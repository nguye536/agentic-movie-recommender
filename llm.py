"""
Movie recommendation agent.

Required environment variables:
  OLLAMA_API_KEY  — injected by grader at runtime

Strategy:
  1. Semantic retrieval  — pre-computed sentence-transformer embeddings loaded
                           from movie_embeddings.npy. Query matched via IDF-weighted
                           pseudo-vector (no embed API call needed at runtime).
                           Falls back to pure TF-IDF if .npy not found.
  2. Popularity re-rank  — blends semantic score (70%) with vote quality (30%)
  3. Rich movie fields   — tagline, director, cast, rating, runtime, keywords
  4. Two-stage LLM       — Stage 1 picks tmdb_id; Stage 2 writes description
  5. Persona-aware tone  — infer user mood and mirror it in the description
"""

import json
import math
import os
import re
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import ollama

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
EMBED_CACHE_PATH = os.path.join(os.path.dirname(__file__), "movie_embeddings.npy")
TOP_K_CANDIDATES = 15

MOOD_MAP = {
    "cozy":      ("cozy, warm, inviting",             ["cozy","comfort","relaxing","feel-good","heartwarming","light","wholesome","family"]),
    "thrilling": ("urgent, electric, propulsive",      ["thriller","suspense","action","adrenaline","intense","tense","edge","gripping","heist"]),
    "emotional": ("emotionally rich, moving",          ["cry","emotional","touching","beautiful","moving","meaningful","deep","feel"]),
    "funny":     ("playful, witty, irreverent",        ["funny","comedy","laugh","humor","raunchy","silly","hilarious","fun","lighthearted"]),
    "epic":      ("grand, sweeping, cinematic",        ["epic","adventure","fantasy","war","hero","quest","journey","world","saga"]),
    "dark":      ("brooding, atmospheric, unsettling", ["dark","disturbing","horror","scary","psychological","gritty","noir","crime","serial"]),
    "thoughtful":("contemplative, intelligent",        ["slow","foreign","art","indie","thoughtful","quiet","nuanced","character","drama"]),
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_df() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="replace")
    xlsx = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.xlsx")
    return pd.read_excel(xlsx)

DF = _load_df()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, maxlen=None) -> str:
    if pd.isna(val): return ""
    s = str(val)
    return s[:maxlen] if maxlen else s

def _build_doc(row) -> str:
    parts = [
        _safe(row.get("title", "")),
        _safe(row.get("tagline", "")),
        _safe(row.get("genres", "")),
        _safe(row.get("overview", ""), 300),
        _safe(row.get("keywords", ""), 200),
        _safe(row.get("director", "")),
        _safe(row.get("top_cast", ""), 100),
    ]
    return " | ".join(p for p in parts if p)

_docs = [_build_doc(row) for _, row in DF.iterrows()]

# ---------------------------------------------------------------------------
# TF-IDF index (pure Python, used for both fallback and query matching)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","it","as","by","be","was","are","this","that","from","i","he","she",
    "they","we","you","my","his","her","their","its","have","has","had","not",
    "film","movie","story","one","two","after","when","who","what","where",
}

def _tokenize(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
            if w not in STOPWORDS and len(w) > 2]

_token_docs = [_tokenize(d) for d in _docs]
_N = len(_token_docs)
_df_counts: dict = defaultdict(int)
for _d in _token_docs:
    for _t in set(_d):
        _df_counts[_t] += 1
_idf: dict = {t: math.log(_N / (c + 1)) for t, c in _df_counts.items()}

def _tfidf_scores(query: str) -> np.ndarray:
    q_tokens = _tokenize(query)
    scores = np.zeros(_N, dtype=np.float32)
    if not q_tokens:
        return scores
    for i, doc in enumerate(_token_docs):
        if not doc:
            continue
        tf: dict = defaultdict(int)
        for t in doc:
            tf[t] += 1
        scores[i] = sum((tf[t] / len(doc)) * _idf.get(t, 0.0) for t in q_tokens if t in tf)
    return scores

# ---------------------------------------------------------------------------
# Semantic retrieval — pre-computed embeddings, no API call at query time
# Query is represented as IDF-weighted average of matching document vectors
# ---------------------------------------------------------------------------

_doc_embeddings: np.ndarray | None = None

if os.path.exists(EMBED_CACHE_PATH):
    _doc_embeddings = np.load(EMBED_CACHE_PATH).astype(np.float32)
    print(f"[embed] loaded {_doc_embeddings.shape[0]} movie embeddings")
else:
    print("[embed] movie_embeddings.npy not found — using TF-IDF")

def _semantic_scores(query: str) -> np.ndarray:
    if _doc_embeddings is None:
        return _tfidf_scores(query)
    # Build query pseudo-vector: IDF-weighted average of doc vectors that
    # share terms with the query. No embed API call needed.
    q_tokens = _tokenize(query)
    if not q_tokens:
        return np.zeros(_N, dtype=np.float32)
    weights = np.zeros(_N, dtype=np.float32)
    for i, doc in enumerate(_token_docs):
        doc_set = set(doc)
        weights[i] = sum(_idf.get(t, 0.0) for t in q_tokens if t in doc_set)
    w_sum = weights.sum()
    if w_sum < 1e-9:
        return _tfidf_scores(query)
    query_vec = (weights[:, None] * _doc_embeddings).sum(axis=0) / w_sum
    norm = np.linalg.norm(query_vec)
    if norm < 1e-9:
        return _tfidf_scores(query)
    query_vec /= norm
    return (_doc_embeddings @ query_vec).astype(np.float32)

# ---------------------------------------------------------------------------
# Quality scores
# ---------------------------------------------------------------------------

def _quality_scores() -> np.ndarray:
    raw = np.array([
        (row.get("vote_average", 0) or 0) * math.log1p(row.get("vote_count", 0) or 0)
        for _, row in DF.iterrows()
    ], dtype=np.float32)
    mn, mx = raw.min(), raw.max()
    return (raw - mn) / (mx - mn + 1e-9)

_qual = _quality_scores()

# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def retrieve_candidates(preferences: str, history_ids: list[int], k: int = TOP_K_CANDIDATES) -> pd.DataFrame:
    sem = _semantic_scores(preferences)
    s_min, s_max = sem.min(), sem.max()
    sem_norm = (sem - s_min) / (s_max - s_min + 1e-9)
    combined = 0.70 * sem_norm + 0.30 * _qual
    history_set = set(history_ids)
    scored = [
        (float(combined[i]), i)
        for i, (_, row) in enumerate(DF.iterrows())
        if int(row["tmdb_id"]) not in history_set
    ]
    scored.sort(reverse=True)
    return DF.iloc[[i for _, i in scored[:k]]].copy()

# ---------------------------------------------------------------------------
# Mood detection
# ---------------------------------------------------------------------------

def detect_mood(preferences: str) -> str:
    prefs_lower = preferences.lower()
    best_mood, best_count = "thoughtful", 0
    for mood, (_, keywords) in MOOD_MAP.items():
        count = sum(1 for kw in keywords if kw in prefs_lower)
        if count > best_count:
            best_mood, best_count = mood, count
    return best_mood

# ---------------------------------------------------------------------------
# Movie card
# ---------------------------------------------------------------------------

def _movie_card(row) -> str:
    parts = [
        f'tmdb_id={int(row["tmdb_id"])}',
        f'"{_safe(row["title"])}" ({int(row["year"]) if pd.notna(row.get("year")) else "?"})',
        f'genres: {_safe(row["genres"])}',
        f'director: {_safe(row["director"])}',
        f'cast: {_safe(row["top_cast"], 80)}',
        f'rating: {_safe(row["us_rating"])} | runtime: {int(row["runtime_min"]) if pd.notna(row.get("runtime_min")) else "?"}min',
        f'score: {float(row["vote_average"]):.1f}/10 ({int(row["vote_count"])} votes)',
    ]
    if pd.notna(row.get("tagline")) and row["tagline"]:
        parts.append(f'tagline: "{_safe(row["tagline"])}"')
    parts.append(f'keywords: {_safe(row["keywords"], 100)}')
    parts.append(f'overview: {_safe(row["overview"], 220)}')
    return "\n    ".join(parts)

# ---------------------------------------------------------------------------
# Ollama client + retry
# ---------------------------------------------------------------------------

DEADLINE_SECS = 18.0   # hard budget; grader kills at 20s

def _get_client() -> ollama.Client:
    import httpx
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        timeout=httpx.Timeout(15.0),
    )

def _call_llm(prompt: str, client: ollama.Client, max_retries: int = 1) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return response.message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"[LLM] retry {attempt+1}/{max_retries} after {wait:.1f}s — {type(e).__name__}")
                time.sleep(wait)
            else:
                raise

# ---------------------------------------------------------------------------
# Stage 1 — pick the best movie
# ---------------------------------------------------------------------------

# Single combined prompt — one LLM call does both selection and description
COMBINED_PROMPT = """\
You are a movie recommendation expert and film critic.

A user wants to watch a movie tonight.

User preferences:
"{preferences}"

Movies they have already seen (DO NOT recommend these):
{history_text}

Candidates:
{movie_list}

Your task:
1. Pick the single best matching movie for this user from the candidates above.
2. Write a description that will make them desperate to watch it tonight.

Description rules:
- Max 490 characters
- Open with a HOOK — not "This film" or "In this movie"
- Mirror the user's own language and emotional register
- Mention 1-2 specific details (actor, scene, directorial choice)
- Tone: {tone_instruction}

Respond ONLY with JSON — no markdown:
{{
  "tmdb_id": <integer>,
  "description": "<your compelling ≤490 char description>"
}}
"""

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    t_start = time.perf_counter()
    client = _get_client()

    candidates = retrieve_candidates(preferences, history_ids)
    valid_ids = set(candidates["tmdb_id"].tolist())
    mood = detect_mood(preferences)
    tone_instruction, _ = MOOD_MAP.get(mood, MOOD_MAP["thoughtful"])

    history_text = (
        ", ".join(f'"{n}" (id={i})' for n, i in zip(history, history_ids))
        if history else "none"
    )
    movie_list = "\n\n".join(
        f"  [{idx+1}]\n    {_movie_card(row)}"
        for idx, (_, row) in enumerate(candidates.iterrows())
    )

    raw = _call_llm(COMBINED_PROMPT.format(
        preferences=preferences,
        history_text=history_text,
        movie_list=movie_list,
        tone_instruction=tone_instruction,
    ), client)

    data = json.loads(raw)
    tmdb_id = int(data["tmdb_id"])
    description = data["description"][:500]

    if tmdb_id not in valid_ids:
        print(f"[warn] id {tmdb_id} not in candidates — using top candidate")
        tmdb_id = int(candidates.iloc[0]["tmdb_id"])

    elapsed = time.perf_counter() - t_start
    print(f"[timing] total: {elapsed:.1f}s")
    return {"tmdb_id": tmdb_id, "description": description}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str)
    args = parser.parse_args()
    preferences = args.preferences.strip() if args.preferences else input("Preferences: ").strip()
    history_raw = args.history.strip() if args.history else input("Watch history (optional): ").strip()
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []
    print(f"\n[mood: {detect_mood(preferences)}] Thinking...\n")
    t0 = time.perf_counter()
    result = get_recommendation(preferences, history)
    movie = DF[DF["tmdb_id"] == result["tmdb_id"]]
    title = movie.iloc[0]["title"] if not movie.empty else "unknown"
    print(f'Recommended: "{title}" (tmdb_id={result["tmdb_id"]})')
    print(f'Description: {result["description"]}')
    print(f'\nServed in {time.perf_counter()-t0:.2f}s')
