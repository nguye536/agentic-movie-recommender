"""Judge results from all 4 llm.py runs and print a comparison table.

Reads results_{hanna,ming,phoebe,yuejia}.jsonl.
Uses ollama cloud to score each recommendation on relevance/quality/specificity/overall.
"""
import json, os, time, sys, re
import pandas as pd
import ollama

def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()

BASE = os.path.dirname(os.path.abspath(__file__))
DF = pd.read_csv(os.path.join(BASE, "tmdb_top1000_movies.csv")).fillna("")
VALID_IDS = set(int(x) for x in DF["tmdb_id"].tolist())

SCRIPTS = ["hanna", "ming", "phoebe", "yuejia"]

def load_results(name):
    path = os.path.join(BASE, f"results_{name}.jsonl")
    out = []
    if not os.path.exists(path):
        return out
    seen_idxs = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" not in obj or obj["idx"] in seen_idxs:
                continue
            seen_idxs.add(obj["idx"])
            out.append(obj)
    out.sort(key=lambda r: r["idx"])
    return out

def lookup(tmdb_id):
    if tmdb_id is None: return None
    row = DF[DF["tmdb_id"] == tmdb_id]
    return row.iloc[0] if not row.empty else None

def dq_check(rec, script_name):
    """Return (ok, reasons) for auto-DQ rules."""
    reasons = []
    if rec.get("error"):
        reasons.append(f"exception: {rec['error']}")
    if rec.get("elapsed", 0) > 20:
        reasons.append(f"over 20s ({rec['elapsed']}s)")
    if rec.get("tmdb_id") is None:
        reasons.append("no tmdb_id returned")
    else:
        if rec["tmdb_id"] not in VALID_IDS:
            reasons.append(f"tmdb_id {rec['tmdb_id']} not in candidate list")
        if rec["tmdb_id"] in rec.get("history_ids", []):
            reasons.append(f"recommended a seen movie (id={rec['tmdb_id']})")
    if rec.get("description") is None:
        reasons.append("no description")
    return (len(reasons) == 0, reasons)

JUDGE_PROMPT = """A user said: "{preferences}"
A recommender suggested: "{title}" ({year}, {genres})
Description: "{description}"

Score 1-5 on each dimension:
- relevance: does this movie match what the user asked for?
- quality: is this a high-quality, well-regarded film?
- specificity: does the description use specific details (cast, plot, themes) vs generic hype?
- overall: would you actually watch this based on the pitch?

Respond with JSON only:
{{"relevance":<1-5>,"quality":<1-5>,"specificity":<1-5>,"overall":<1-5>,"reasoning":"<one sentence>"}}
"""

def get_judge_client():
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )

def judge_one(client, preferences, row, description, retries=2):
    prompt = JUDGE_PROMPT.format(
        preferences=preferences,
        title=row["title"],
        year=int(row.get("year", 0) or 0),
        genres=row.get("genres", ""),
        description=description,
    )
    for attempt in range(retries):
        try:
            resp = client.chat(
                model="gemma4:31b-cloud",
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return json.loads(_strip_fences(resp.message.content))
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"relevance": 0, "quality": 0, "specificity": 0, "overall": 0, "reasoning": f"judge error: {e}"}

def main():
    # Load all results
    results = {s: load_results(s) for s in SCRIPTS}

    # DQ summary
    print("=" * 90)
    print("AUTO-DISQUALIFICATION CHECK")
    print("=" * 90)
    dq_counts = {s: 0 for s in SCRIPTS}
    dq_details = {s: [] for s in SCRIPTS}
    for s in SCRIPTS:
        for rec in results[s]:
            ok, reasons = dq_check(rec, s)
            if not ok:
                dq_counts[s] += 1
                dq_details[s].append((rec["idx"], rec["preferences"][:45], reasons))
        print(f"\n{s.upper()}: {dq_counts[s]}/20 would DQ")
        for idx, pref, reasons in dq_details[s][:10]:
            print(f"  [{idx:2d}] \"{pref}\" -> {'; '.join(reasons)}")

    # Timing summary
    print("\n" + "=" * 90)
    print("TIMING")
    print("=" * 90)
    for s in SCRIPTS:
        times = [r["elapsed"] for r in results[s] if r.get("elapsed") is not None]
        if times:
            print(f"{s.upper():10s}  min={min(times):5.1f}s  max={max(times):5.1f}s  avg={sum(times)/len(times):5.1f}s  over_20s={sum(1 for t in times if t > 20)}")

    # Judge non-DQ'd recommendations
    print("\n" + "=" * 90)
    print("LLM-AS-JUDGE SCORING (only on non-DQ'd recommendations)")
    print("=" * 90)
    client = get_judge_client()
    scored = {s: [] for s in SCRIPTS}

    n_prompts = max(len(results[s]) for s in SCRIPTS) if any(results.values()) else 0
    for i in range(n_prompts):
        prompt_text = None
        for s in SCRIPTS:
            if i < len(results[s]):
                prompt_text = results[s][i]["preferences"]
                break
        if not prompt_text:
            continue
        print(f"\n[{i+1:2d}/{n_prompts}] \"{prompt_text[:55]}\"")
        for s in SCRIPTS:
            if i >= len(results[s]):
                print(f"  {s:10s}: (no result)")
                continue
            rec = results[s][i]
            ok, reasons = dq_check(rec, s)
            if not ok:
                print(f"  {s:10s}: DQ ({'; '.join(reasons)[:60]})")
                scored[s].append(None)
                continue
            row = lookup(rec["tmdb_id"])
            if row is None:
                print(f"  {s:10s}: tmdb_id not found in DF")
                scored[s].append(None)
                continue
            scores = judge_one(client, rec["preferences"], row, rec["description"])
            scored[s].append(scores)
            print(f"  {s:10s}: \"{row['title']}\" rel={scores.get('relevance')} qual={scores.get('quality')} spec={scores.get('specificity')} overall={scores.get('overall')}")

    # Save raw
    with open(os.path.join(BASE, "judge_output.json"), "w") as f:
        json.dump({"scored": scored, "dq_counts": dq_counts}, f, indent=2)

    # Aggregate
    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)
    print(f"{'script':10s}  {'DQ':>4s}  {'rel':>5s}  {'qual':>5s}  {'spec':>5s}  {'overall':>7s}")
    print("-" * 60)
    for s in SCRIPTS:
        valid = [x for x in scored[s] if x is not None]
        if not valid:
            print(f"{s:10s}  {dq_counts[s]:>4d}  {'--':>5s}  {'--':>5s}  {'--':>5s}  {'--':>7s}")
            continue
        rel = sum(x.get("relevance", 0) for x in valid) / len(valid)
        qual = sum(x.get("quality", 0) for x in valid) / len(valid)
        spec = sum(x.get("specificity", 0) for x in valid) / len(valid)
        overall = sum(x.get("overall", 0) for x in valid) / len(valid)
        print(f"{s:10s}  {dq_counts[s]:>4d}  {rel:>5.2f}  {qual:>5.2f}  {spec:>5.2f}  {overall:>7.2f}")

if __name__ == "__main__":
    main()
