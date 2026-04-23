"""Quick summary of results — DQ check + timing + sample recommendations, no LLM judge."""
import json, os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DF = pd.read_csv(os.path.join(BASE, "tmdb_top1000_movies.csv")).fillna("")
VALID_IDS = set(int(x) for x in DF["tmdb_id"].tolist())

SCRIPTS = ["hanna", "ming", "phoebe", "yuejia"]

def load(name):
    path = os.path.join(BASE, f"results_{name}.jsonl")
    out = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try: obj = json.loads(line)
            except: continue
            if "idx" not in obj or obj["idx"] in seen: continue
            seen.add(obj["idx"])
            out.append(obj)
    out.sort(key=lambda r: r["idx"])
    return out

def title_for(tid):
    if tid is None: return "?"
    row = DF[DF["tmdb_id"] == tid]
    return row.iloc[0]["title"] if not row.empty else f"[id {tid} not in DF]"

def dq(rec):
    r = []
    if rec.get("error"): r.append(f"ERR:{rec['error'][:40]}")
    if rec.get("elapsed", 0) > 20: r.append(f"OVER20s ({rec['elapsed']}s)")
    if rec.get("tmdb_id") is None: r.append("no_id")
    elif rec["tmdb_id"] not in VALID_IDS: r.append(f"invalid_id {rec['tmdb_id']}")
    elif rec["tmdb_id"] in rec.get("history_ids", []): r.append("seen_movie")
    if rec.get("description") is None: r.append("no_desc")
    return r

results = {s: load(s) for s in SCRIPTS}

print("="*95)
print("PER-PROMPT RESULTS")
print("="*95)
for i in range(20):
    if not all(i < len(results[s]) for s in SCRIPTS): continue
    prompt = results["hanna"][i]["preferences"]
    print(f"\n[{i+1:2d}] \"{prompt}\"")
    for s in SCRIPTS:
        rec = results[s][i]
        title = title_for(rec.get("tmdb_id"))
        issues = dq(rec)
        tag = f"  {'DQ:'+','.join(issues) if issues else ''}"
        print(f"  {s:7s} {rec['elapsed']:5.1f}s  \"{title[:38]:38s}\"{tag}")

print("\n" + "="*95)
print("SUMMARY")
print("="*95)
print(f"{'script':10s} {'DQ':>4s} {'min':>5s} {'max':>5s} {'avg':>5s} {'>20s':>5s} {'extra_keys':>12s}")
print("-"*60)
for s in SCRIPTS:
    recs = results[s]
    times = [r["elapsed"] for r in recs]
    dq_count = sum(1 for r in recs if dq(r))
    over_20 = sum(1 for t in times if t > 20)
    extras = sum(1 for r in recs if r.get("extra_keys"))
    print(f"{s:10s} {dq_count:>4d} {min(times):>5.1f} {max(times):>5.1f} {sum(times)/len(times):>5.1f} {over_20:>5d} {extras:>12d}")

print("\n" + "="*95)
print("DQ DETAILS")
print("="*95)
for s in SCRIPTS:
    issues = [(r["idx"], dq(r)) for r in results[s] if dq(r)]
    if not issues:
        print(f"{s:10s}: no DQ issues")
        continue
    print(f"{s:10s}: {len(issues)} DQ events")
    for idx, reasons in issues:
        prompt = results[s][idx]["preferences"]
        print(f"  [{idx:2d}] \"{prompt[:50]}\" -> {', '.join(reasons)}")

print("\n" + "="*95)
print("SAMPLE DESCRIPTIONS (prompt 3: 'I want to cry in a good way')")
print("="*95)
for s in SCRIPTS:
    rec = results[s][2]
    print(f"\n{s.upper()}: \"{title_for(rec.get('tmdb_id'))}\"")
    print(f"  {rec.get('description', '')[:450]}")

print("\n" + "="*95)
print("SAMPLE DESCRIPTIONS (prompt 15: 'something by Christopher Nolan')")
print("="*95)
for s in SCRIPTS:
    rec = results[s][13]
    print(f"\n{s.upper()}: \"{title_for(rec.get('tmdb_id'))}\"")
    print(f"  {rec.get('description', '')[:450]}")
