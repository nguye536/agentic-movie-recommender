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
from functools import lru_cache

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

    # Director/actor boost: if preferences mention a name that matches director or cast, boost those films
    prefs_lower = preferences.lower()
    director_boost = np.zeros(len(DF), dtype=np.float32)
    cast_boost = np.zeros(len(DF), dtype=np.float32)
    for i, (_, row) in enumerate(DF.iterrows()):
        director = _safe(row.get("director", "")).lower()
        cast = _safe(row.get("top_cast", "")).lower()
        # Check if any word 4+ chars from preferences matches director or cast
        for word in re.findall(r"[a-z]{4,}", prefs_lower):
            if word in director:
                director_boost[i] = 0.4
            if word in cast:
                cast_boost[i] = 0.2

    combined = combined + director_boost + cast_boost

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
        f'cast: {_safe(row["top_cast"], 60)}',
    ]
    if pd.notna(row.get("tagline")) and row["tagline"]:
        parts.append(f'tagline: "{_safe(row["tagline"])}"')
    parts.append(f'overview: {_safe(row["overview"], 150)}')
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
        timeout=httpx.Timeout(19.0),
    )

def _call_llm(prompt: str, client: ollama.Client, max_retries: int = 2) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            content = response.message.content
            # Retry if empty response
            if not content or not content.strip():
                raise ValueError("empty response from LLM")
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 1.0 + random.uniform(0, 0.5)
                print(f"[LLM] retry {attempt+1}/{max_retries} after {wait:.1f}s — {e}")
                time.sleep(wait)
            else:
                raise

# ---------------------------------------------------------------------------
# Stage 1 — pick the best movie
# ---------------------------------------------------------------------------

# Single combined prompt — one LLM call does both selection and description
COMBINED_PROMPT = """\
You recommend movies like a knowledgeable friend — warm and specific, not a critic, not a salesperson.

A user wants to watch something tonight:
"{preferences}"

Already seen (do not recommend):
{history_text}

Candidates:
{movie_list}

Write a 2-3 sentence recommendation. The goal is to make them genuinely excited, not to sell them something.

RULES:
- Max 460 characters. Always end on a complete sentence.
- Sentence 1: lead with the feeling or the hook — what makes this movie special for THIS person.
- Sentence 2: one specific, concrete detail — a scene, a performance, a twist — that brings it to life.
- Sentence 3: a short, natural closer that ties back to what they said they want. Warm but not pushy.
- Write like a person, not a review. Avoid: "masterpiece", "stunning", "visceral", "cerebral", "you will love", "perfect for you", "put it on".
- Tone: {tone_instruction}

Respond ONLY with JSON:
{{
  "tmdb_id": <integer>,
  "description": "<your 2-3 sentence ≤460 char recommendation>"
}}
"""

# In-memory cache — avoids repeat LLM calls for identical inputs within a session
_response_cache: dict = {}

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    # Cache lookup — same preferences + history returns instantly
    cache_key = (preferences.strip().lower(), tuple(sorted(history_ids)))
    if cache_key in _response_cache:
        print("[cache] hit — returning cached result")
        return _response_cache[cache_key]

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

    # Guard against empty or malformed LLM response
    raw = (raw or "").strip()
    if not raw:
        print("[warn] empty LLM response — using top candidate with fallback description")
        tmdb_id = int(candidates.iloc[0]["tmdb_id"])
        row = candidates.iloc[0]
        description = f"{_safe(row['title'])}: {_safe(row['overview'], 400)}"[:500]
        return {"tmdb_id": tmdb_id, "description": description}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON from response if wrapped in text
        import re as _re
        match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            print(f"[warn] could not parse LLM response: {raw[:100]}")
            tmdb_id = int(candidates.iloc[0]["tmdb_id"])
            row = candidates.iloc[0]
            description = f"{_safe(row['title'])}: {_safe(row['overview'], 400)}"[:500]
            return {"tmdb_id": tmdb_id, "description": description}

    tmdb_id = int(data["tmdb_id"])
    description = data["description"][:500]

    if tmdb_id not in valid_ids:
        print(f"[warn] id {tmdb_id} not in candidates — using top candidate")
        tmdb_id = int(candidates.iloc[0]["tmdb_id"])

    elapsed = time.perf_counter() - t_start
    print(f"[timing] total: {elapsed:.1f}s")
    result = {"tmdb_id": tmdb_id, "description": description}
    _response_cache[cache_key] = result
    return result

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