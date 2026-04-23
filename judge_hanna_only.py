"""Re-judge just Hanna's results after DESCRIBE_PROMPT changes."""
import json, os, time, re
import pandas as pd
import ollama
from dotenv import load_dotenv
load_dotenv()

def _strip_fences(s):
    s = (s or "").strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```\s*$', '', s)
    return s.strip()

BASE = os.path.dirname(os.path.abspath(__file__))
DF = pd.read_csv(os.path.join(BASE, "tmdb_top1000_movies.csv")).fillna("")
VALID_IDS = set(int(x) for x in DF["tmdb_id"].tolist())

def load(name):
    out, seen = [], set()
    with open(os.path.join(BASE, f"results_{name}.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"): continue
            try: obj = json.loads(line)
            except: continue
            if "idx" not in obj or obj["idx"] in seen: continue
            seen.add(obj["idx"]); out.append(obj)
    out.sort(key=lambda r: r["idx"])
    return out

def lookup(tid):
    row = DF[DF["tmdb_id"] == tid]
    return row.iloc[0] if not row.empty else None

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

client = ollama.Client(host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"})

def judge_one(pref, row, desc):
    prompt = JUDGE_PROMPT.format(preferences=pref, title=row["title"],
        year=int(row.get("year", 0) or 0), genres=row.get("genres", ""), description=desc)
    for attempt in range(3):
        try:
            resp = client.chat(model="gemma4:31b-cloud",
                messages=[{"role":"user","content":prompt}], format="json")
            return json.loads(_strip_fences(resp.message.content))
        except Exception as e:
            if attempt < 2: time.sleep(2)
            else: return {"relevance":0,"quality":0,"specificity":0,"overall":0,"reasoning":f"err: {e}"}

recs = load("hanna")
scored = []
for rec in recs:
    row = lookup(rec["tmdb_id"])
    if row is None:
        scored.append(None); continue
    s = judge_one(rec["preferences"], row, rec["description"])
    scored.append(s)
    print(f"[{rec['idx']:2d}] {row['title'][:35]:35s}  rel={s.get('relevance')} qual={s.get('quality')} spec={s.get('specificity')} overall={s.get('overall')}")

valid = [x for x in scored if x is not None]
rel = sum(x.get("relevance",0) for x in valid)/len(valid)
qual = sum(x.get("quality",0) for x in valid)/len(valid)
spec = sum(x.get("specificity",0) for x in valid)/len(valid)
overall = sum(x.get("overall",0) for x in valid)/len(valid)
print(f"\nHANNA (updated)  rel={rel:.2f}  qual={qual:.2f}  spec={spec:.2f}  overall={overall:.2f}")

with open(os.path.join(BASE, "judge_hanna_updated.json"), "w") as f:
    json.dump(scored, f, indent=2)
