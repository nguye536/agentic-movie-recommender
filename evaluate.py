"""
eval/evaluate.py — evaluation suite for the movie recommender.

Usage:
  python eval/evaluate.py --mode timing   # check all 20 prompts stay under 20s
  python eval/evaluate.py --mode score    # LLM-as-judge scores all 20 prompts
  python eval/evaluate.py --mode ab       # A/B vs naive baseline
  python eval/evaluate.py --mode quick    # interactive single-prompt eval

Requires: OLLAMA_API_KEY in environment
"""

import json, os, time, argparse, statistics

from llm import get_recommendation, DF, detect_mood, _client, MODEL, _semantic_scores, retrieve_candidates

TEST_PROMPTS = [
    {"preferences": "something good for tonight",                                        "history": [], "history_ids": []},
    {"preferences": "I don't know, just surprise me",                                    "history": [], "history_ids": []},
    {"preferences": "I want to cry in a good way",                                       "history": [], "history_ids": []},
    {"preferences": "something cozy and comforting for a rainy Sunday",                  "history": [], "history_ids": []},
    {"preferences": "I need adrenaline — edge-of-seat stuff",                            "history": [], "history_ids": []},
    {"preferences": "something funny and lighthearted, had a rough week",                "history": [], "history_ids": []},
    {"preferences": "a slow-burn psychological thriller",                                 "history": [], "history_ids": []},
    {"preferences": "a raunchy comedy like Superbad or The Hangover",                    "history": [], "history_ids": []},
    {"preferences": "an epic fantasy adventure with great world-building",                "history": [], "history_ids": []},
    {"preferences": "a gritty crime drama, something like The Wire",                     "history": [], "history_ids": []},
    {"preferences": "a foreign language film with stunning cinematography",               "history": [], "history_ids": []},
    {"preferences": "an indie character study, quiet and nuanced",                       "history": [], "history_ids": []},
    {"preferences": "an animated film that adults will love too",                         "history": [], "history_ids": []},
    {"preferences": "something by Christopher Nolan",                                     "history": [], "history_ids": []},
    {"preferences": "a film with Margot Robbie",                                         "history": [], "history_ids": []},
    {"preferences": "I love Marvel, want more superhero action",
     "history": ["The Avengers", "Avengers: Infinity War"], "history_ids": [24428, 299536]},
    {"preferences": "dark comedy with a twist ending",
     "history": ["Parasite", "Joker"],                      "history_ids": [496243, 475557]},
    {"preferences": "something romantic but not cheesy",                                 "history": [], "history_ids": []},
    {"preferences": "action but also thoughtful and quiet",                              "history": [], "history_ids": []},
    {"preferences": "scary but not too scary, no gore",                                  "history": [], "history_ids": []},
]

def _call_judge(prompt, retries=2):
    for attempt in range(retries):
        try:
            resp = _client.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], format="json")
            return json.loads(resp.message.content)
        except Exception as e:
            if attempt < retries - 1: time.sleep(2)
            else: raise

# ---------------------------------------------------------------------------
# Timing mode
# ---------------------------------------------------------------------------

def _sem_score(preferences, rec_tmdb_id, history_ids):
    """Return the semantic score (0-1) of the recommended movie vs all candidates."""
    sem = _semantic_scores(preferences)
    s_min, s_max = sem.min(), sem.max()
    sem_norm = (sem - s_min) / (s_max - s_min + 1e-9)
    idx = DF[DF["tmdb_id"] == rec_tmdb_id].index
    return float(sem_norm[idx[0]]) if len(idx) else 0.0

def run_timing():
    print(f"=== Timing Test ({len(TEST_PROMPTS)} prompts, limit: 20s) ===\n")
    times, failures = [], []
    for i, p in enumerate(TEST_PROMPTS):
        print(f"[{i+1:2d}/{len(TEST_PROMPTS)}] \"{p['preferences'][:55]}\"", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            rec = get_recommendation(p["preferences"], p["history"], p["history_ids"])
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            ok = elapsed < 20
            sem = _sem_score(p["preferences"], rec["tmdb_id"], p["history_ids"])
            print(f"→ {elapsed:.1f}s  sem={sem:.2f}  {'✅' if ok else '❌ OVER LIMIT'}")
            if not ok: failures.append((p["preferences"], elapsed))
        except Exception as e:
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"→ {elapsed:.1f}s ❌ ERROR: {e}")
            failures.append((p["preferences"], elapsed))

    print(f"\n=== Results ===")
    print(f"  Min:    {min(times):.1f}s")
    print(f"  Max:    {max(times):.1f}s")
    print(f"  Avg:    {statistics.mean(times):.1f}s")
    print(f"  Median: {statistics.median(times):.1f}s")
    print(f"  Passed: {len(times)-len(failures)}/{len(times)}")
    if failures:
        print(f"\n❌ Failed:")
        for pref, t in failures: print(f"  {t:.1f}s — \"{pref[:50]}\"")
    else:
        print(f"\n✅ All prompts under 20s!")

# ---------------------------------------------------------------------------
# Score mode
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
A user said: "{preferences}"
A recommender suggested: "{title}" ({year}, {genres})
Description: "{description}"

Score 1-5 on: relevance, quality, specificity, overall.
JSON only: {{"relevance":<1-5>,"quality":<1-5>,"specificity":<1-5>,"overall":<1-5>,"reasoning":"<one sentence>"}}
"""

def run_score():
    results, totals, n = [], {"relevance":0,"quality":0,"specificity":0,"overall":0}, 0
    print(f"=== LLM-as-Judge ({len(TEST_PROMPTS)} prompts) ===\n")
    for i, p in enumerate(TEST_PROMPTS):
        print(f"[{i+1:2d}/{len(TEST_PROMPTS)}] \"{p['preferences'][:50]}\"")
        try:
            rec = get_recommendation(p["preferences"], p["history"], p["history_ids"])
            movie = DF[DF["tmdb_id"] == rec["tmdb_id"]]
            if movie.empty: print("  [skip]"); continue
            row = movie.iloc[0]
            sem = _sem_score(p["preferences"], rec["tmdb_id"], p["history_ids"])
            scores = _call_judge(JUDGE_PROMPT.format(preferences=p["preferences"],
                title=row["title"], year=int(row.get("year",0)),
                genres=row["genres"], description=rec["description"]))
            results.append({**p, "recommended": row["title"], "semantic_score": round(sem, 3), "scores": scores})
            for k in totals: totals[k] += scores.get(k, 0)
            n += 1
            print(f"  → {row['title']} | sem={sem:.2f} | rel={scores.get('relevance')} qual={scores.get('quality')} spec={scores.get('specificity')} overall={scores.get('overall')}")
            print(f"     {scores.get('reasoning','')}")
        except Exception as e:
            print(f"  [error] {e}")
    if n:
        print(f"\n=== Averages (n={n}) ===")
        for k, v in totals.items(): print(f"  {k}: {v/n:.2f}/5")
    os.makedirs("eval", exist_ok=True)
    with open("eval/results_score.json","w") as f:
        json.dump({"results":results,"averages":{k:round(v/max(n,1),2) for k,v in totals.items()}},f,indent=2)
    print("Saved to eval/results_score.json")

# ---------------------------------------------------------------------------
# A/B mode
# ---------------------------------------------------------------------------

AB_PROMPT = """\
A user wants: "{preferences}"
A: "{title_a}" — "{desc_a}"
B: "{title_b}" — "{desc_b}"
Which would you rather watch? JSON: {{"winner":"A" or "B" or "tie","reasoning":"<one sentence>"}}
"""

def run_ab():
    def baseline(preferences, history, history_ids):
        pool = DF[~DF["tmdb_id"].isin(history_ids)].nlargest(5, "vote_count")
        row = pool.iloc[0]
        return {"tmdb_id": int(row["tmdb_id"]), "description": str(row["overview"])[:500]}

    wins, results = {"A":0,"B":0,"tie":0}, []
    print(f"=== A/B: Our agent (A) vs Baseline (B) ===\n")
    for i, p in enumerate(TEST_PROMPTS):
        print(f"[{i+1:2d}/{len(TEST_PROMPTS)}] \"{p['preferences'][:50]}\"")
        try:
            rec_a = get_recommendation(p["preferences"], p["history"], p["history_ids"])
            rec_b = baseline(p["preferences"], p["history"], p["history_ids"])
            ma = DF[DF["tmdb_id"]==rec_a["tmdb_id"]].iloc[0]
            mb = DF[DF["tmdb_id"]==rec_b["tmdb_id"]].iloc[0]
            verdict = _call_judge(AB_PROMPT.format(preferences=p["preferences"],
                title_a=ma["title"], desc_a=rec_a["description"],
                title_b=mb["title"], desc_b=rec_b["description"]))
            w = verdict.get("winner","tie")
            wins[w] = wins.get(w,0)+1
            results.append({"prompt":p["preferences"],"a":ma["title"],"b":mb["title"],"winner":w,"reasoning":verdict.get("reasoning","")})
            print(f"  {ma['title']} vs {mb['title']} → {w}")
        except Exception as e:
            print(f"  [error] {e}")
    print(f"\n=== Results ===\n  Our wins: {wins['A']}  Baseline: {wins['B']}  Ties: {wins.get('tie',0)}")
    os.makedirs("eval", exist_ok=True)
    with open("eval/results_ab.json","w") as f:
        json.dump({"wins":wins,"results":results},f,indent=2)
    print("Saved to eval/results_ab.json")

# ---------------------------------------------------------------------------
# Quick mode
# ---------------------------------------------------------------------------

def run_quick():
    preferences = input("Preferences: ").strip()
    history_raw = input("History (optional, comma-separated): ").strip()
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []
    print(f"\n[mood: {detect_mood(preferences)}] Thinking...\n")
    t0 = time.perf_counter()
    result = get_recommendation(preferences, history)
    elapsed = time.perf_counter() - t0
    movie = DF[DF["tmdb_id"] == result["tmdb_id"]]
    title = movie.iloc[0]["title"] if not movie.empty else "unknown"
    print(f'Recommended: "{title}" (tmdb_id={result["tmdb_id"]})')
    print(f'Description: {result["description"]}')
    print(f'Time: {elapsed:.1f}s {"✅" if elapsed < 20 else "❌ OVER LIMIT"}')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["score","ab","timing","quick"], default="quick")
    args = parser.parse_args()
    {"score": run_score, "ab": run_ab, "timing": run_timing, "quick": run_quick}[args.mode]()
