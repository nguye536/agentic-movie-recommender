"""
Movie recommendation agent — improved implementation.

Required environment variables:
  OLLAMA_API_KEY   — injected by grader at runtime

Optional environment variables:
  USE_TFIDF=1      — force TF-IDF retrieval even if sentence-transformers is available
                     (useful for fast local testing; omit in production)

Strategy overview:
  1. Semantic retrieval  — embed all 720 movies; retrieve top-15 by cosine similarity
                           (falls back to TF-IDF if sentence-transformers unavailable)
  2. Popularity re-rank  — blend semantic score with vote quality score before LLM call
  3. Rich movie fields   — tagline, director, cast, rating, runtime, keywords all in prompt
  4. Two-stage LLM       — Stage 1 picks the best tmdb_id; Stage 2 writes the description
  5. Persona-aware tone  — infer user mood and mirror it in the description
"""

import json
import os
import time
import random
import re
import argparse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import ollama

# Optional ML imports — lazy load to handle deployment environments
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")
TOP_K_CANDIDATES = 15   # candidates passed to LLM
EMBED_CACHE_PATH = os.path.join(os.path.dirname(__file__), "movie_embeddings.npy")

# Mood vocabulary → tone instruction for Stage 2 prompt
MOOD_MAP = {
    "cozy":        ("cozy, warm, inviting",       ["cozy","comfort","relaxing","feel-good","heartwarming","light","wholesome","family"]),
    "thrilling":   ("urgent, electric, propulsive",["thriller","suspense","action","adrenaline","intense","tense","edge","gripping","heist"]),
    "emotional":   ("emotionally rich, moving",   ["cry","emotional","touching","beautiful","moving","meaningful","deep","feel"]),
    "funny":       ("playful, witty, irreverent",  ["funny","comedy","laugh","humor","raunchy","silly","hilarious","fun","lighthearted"]),
    "epic":        ("grand, sweeping, cinematic",  ["epic","adventure","fantasy","war","hero","quest","journey","world","saga"]),
    "dark":        ("brooding, atmospheric, unsettling", ["dark","disturbing","horror","scary","psychological","gritty","noir","crime","serial"]),
    "thoughtful":  ("contemplative, intelligent",  ["slow","foreign","art","indie","thoughtful","quiet","nuanced","character","drama"]),
}

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_df() -> pd.DataFrame:
    """Load from CSV (competition) or fall back to xlsx (development)."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        xlsx_path = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.xlsx")
        df = pd.read_excel(xlsx_path)
    return df

DF = _load_df()

# ---------------------------------------------------------------------------
# Retrieval — semantic + TF-IDF fallback
# ---------------------------------------------------------------------------

def _safe(val, maxlen=None) -> str:
    if pd.isna(val):
        return ""
    s = str(val)
    return s[:maxlen] if maxlen else s


def _build_embed_text(row) -> str:
    parts = [
        _safe(row.get("title", "")),
        _safe(row.get("tagline", "")),
        _safe(row.get("genres", "")),
        _safe(row.get("overview", ""), 300),
        _safe(row.get("keywords", ""), 150),
        _safe(row.get("director", "")),
        _safe(row.get("top_cast", ""), 100),
    ]
    return " | ".join(p for p in parts if p)


_embed_texts = [_build_embed_text(row) for _, row in DF.iterrows()]

# Try loading sentence-transformers (fast semantic model)
_st_model = None
_st_embeddings = None

if not os.environ.get("USE_TFIDF"):
    try:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        if os.path.exists(EMBED_CACHE_PATH):
            _st_embeddings = np.load(EMBED_CACHE_PATH)
        else:
            print("Building movie embeddings (one-time, ~30s)...")
            _st_embeddings = _st_model.encode(
                _embed_texts, batch_size=64, show_progress_bar=False
            )
            np.save(EMBED_CACHE_PATH, _st_embeddings)
            print("Embeddings cached to", EMBED_CACHE_PATH)
    except Exception as e:
        print(f"[retrieval] sentence-transformers unavailable ({e}), using TF-IDF")

# TF-IDF fallback (always built; used if sentence-transformers failed)
_tfidf_vec = None
_tfidf_matrix = None

if HAS_SKLEARN:
    _tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=10000)
    _tfidf_matrix = _tfidf_vec.fit_transform(_embed_texts)


def _semantic_scores(query: str) -> np.ndarray:
    """Return per-movie cosine similarity scores for the query string."""
    if _st_model is not None and _st_embeddings is not None:
        q_emb = _st_model.encode([query])
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(q_emb, _st_embeddings)[0]
    elif HAS_SKLEARN and _tfidf_vec is not None and _tfidf_matrix is not None:
        q_vec = _tfidf_vec.transform([query])
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(q_vec, _tfidf_matrix)[0]
    else:
        # Fallback: simple keyword matching when sklearn unavailable
        query_lower = query.lower()
        scores = np.array([
            sum(1 for word in query_lower.split() if word in _safe(row.get("overview", "")).lower())
            for _, row in DF.iterrows()
        ], dtype=float)
        return scores / (np.max(scores) + 1e-9) if np.max(scores) > 0 else scores


def _quality_scores(df: pd.DataFrame) -> np.ndarray:
    """Normalised vote_average × log(vote_count) quality signal."""
    raw = df["vote_average"].fillna(0) * np.log1p(df["vote_count"].fillna(0))
    mn, mx = raw.min(), raw.max()
    return ((raw - mn) / (mx - mn + 1e-9)).values


def retrieve_candidates(
    preferences: str, history_ids: list[int], k: int = TOP_K_CANDIDATES
) -> pd.DataFrame:
    """Return top-k candidate movies, excluding already-seen ones."""
    mask = ~DF["tmdb_id"].isin(history_ids)
    pool = DF[mask].copy()
    pool_idx = pool.index.tolist()

    sem = _semantic_scores(preferences)
    qual = _quality_scores(DF)

    # Blend: 70% semantic relevance, 30% quality
    combined = 0.70 * sem + 0.30 * qual
    pool_scores = [(combined[i], i) for i in pool_idx]
    pool_scores.sort(key=lambda x: x[0], reverse=True)

    top_indices = [i for _, i in pool_scores[:k]]
    return DF.loc[top_indices].copy()

# ---------------------------------------------------------------------------
# Mood detection
# ---------------------------------------------------------------------------

def detect_mood(preferences: str) -> str:
    """Return a mood key from MOOD_MAP based on keyword matching."""
    prefs_lower = preferences.lower()
    best_mood, best_count = "thoughtful", 0
    for mood, (_, keywords) in MOOD_MAP.items():
        count = sum(1 for kw in keywords if kw in prefs_lower)
        if count > best_count:
            best_mood, best_count = mood, count
    return best_mood

# ---------------------------------------------------------------------------
# Movie card for prompt
# ---------------------------------------------------------------------------

def _movie_card(row) -> str:
    parts = [
        f'tmdb_id={int(row["tmdb_id"])}',
        f'"{_safe(row["title"])}" ({int(row["year"]) if pd.notna(row.get("year")) else "?"})',
        f'genres: {_safe(row["genres"])}',
        f'director: {_safe(row["director"])}',
        f'cast: {_safe(row["top_cast"], 80)}',
        f'rating: {_safe(row["us_rating"])} | runtime: {int(row["runtime_min"]) if pd.notna(row.get("runtime_min")) else "?"}min',
        f'score: {row["vote_average"]:.1f}/10 ({int(row["vote_count"])} votes)',
    ]
    if pd.notna(row.get("tagline")) and row["tagline"]:
        parts.append(f'tagline: "{_safe(row["tagline"])}"')
    parts.append(f'keywords: {_safe(row["keywords"], 100)}')
    parts.append(f'overview: {_safe(row["overview"], 220)}')
    return "\n    ".join(parts)

# ---------------------------------------------------------------------------
# Ollama client + retry
# ---------------------------------------------------------------------------

def _get_client() -> ollama.Client:
    return ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )


def _call_llm(prompt: str, client: ollama.Client, max_retries: int = 3) -> str:
    """Call the LLM with exponential backoff. Returns raw message content."""
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

STAGE1_PROMPT = """\
You are a movie recommendation expert. A user wants to watch a movie tonight.

User preferences:
"{preferences}"

Movies they have already seen (DO NOT recommend these):
{history_text}

Below are {k} candidate movies. Your job is to pick the single best match for this user.

Think step by step before deciding:
1. What does this user REALLY want emotionally and thematically?
2. Which movie best fits their specific preferences — not just surface genre?
3. Are there any obvious mismatches to rule out?

Candidates:
{movie_list}

Respond ONLY with a JSON object — no markdown, no extra text:
{{
  "tmdb_id": <integer>,
  "reasoning": "<2-3 sentences explaining why this is the best pick>"
}}
"""

def stage1_select(
    preferences: str,
    history: list[str],
    history_ids: list[int],
    candidates: pd.DataFrame,
    client: ollama.Client,
) -> tuple[int, str]:
    """Returns (tmdb_id, reasoning)."""
    history_text = (
        ", ".join(f'"{n}" (id={i})' for n, i in zip(history, history_ids))
        if history else "none"
    )
    movie_list = "\n\n".join(
        f"  [{idx+1}]\n    {_movie_card(row)}"
        for idx, (_, row) in enumerate(candidates.iterrows())
    )
    prompt = STAGE1_PROMPT.format(
        preferences=preferences,
        history_text=history_text,
        k=len(candidates),
        movie_list=movie_list,
    )
    raw = _call_llm(prompt, client)
    data = json.loads(raw)
    return int(data["tmdb_id"]), data.get("reasoning", "")

# ---------------------------------------------------------------------------
# Stage 2 — write a killer description
# ---------------------------------------------------------------------------

STAGE2_PROMPT = """\
You are a brilliant film critic writing for an audience of movie lovers.

A user is looking for a movie to watch. Here is what they said:
"{preferences}"

You have selected this movie for them:
{movie_card}

Your reasoning for selecting it:
{reasoning}

Now write a description that will make this user desperate to watch the film tonight.

Rules:
- Max 500 characters (it will be truncated if longer — aim for 460-490 chars to be safe)
- Open with a HOOK — not "This film" or "In this movie". Start with emotion, action, or intrigue.
- Mirror the user's own language and emotional register. If they said "{mood_word}", echo that energy.
- Mention 1-2 specific details (a scene, actor, directorial choice) that match their preferences.
- End with a reason they personally — given what they said — will love it.
- Tone: {tone_instruction}

Respond ONLY with JSON — no markdown:
{{
  "description": "<your ≤500 char compelling description>"
}}
"""

def stage2_describe(
    preferences: str,
    tmdb_id: int,
    reasoning: str,
    mood: str,
    candidates: pd.DataFrame,
    client: ollama.Client,
) -> str:
    """Returns the final description string."""
    row = candidates[candidates["tmdb_id"] == tmdb_id].iloc[0]
    tone_instruction, _ = MOOD_MAP.get(mood, MOOD_MAP["thoughtful"])

    # Extract a mood word from preferences to echo back
    prefs_lower = preferences.lower()
    _, keywords = MOOD_MAP.get(mood, ("", []))
    mood_word = next((kw for kw in keywords if kw in prefs_lower), mood)

    prompt = STAGE2_PROMPT.format(
        preferences=preferences,
        movie_card=_movie_card(row),
        reasoning=reasoning,
        mood_word=mood_word,
        tone_instruction=tone_instruction,
    )
    raw = _call_llm(prompt, client)
    data = json.loads(raw)
    return data["description"][:500]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendation(
    preferences: str, history: list[str], history_ids: list[int] = []
) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str)."""
    client = _get_client()

    # 1. Retrieve candidates
    candidates = retrieve_candidates(preferences, history_ids)
    valid_ids = set(candidates["tmdb_id"].tolist())

    # 2. Detect mood
    mood = detect_mood(preferences)

    # 3. Stage 1 — pick the movie
    tmdb_id, reasoning = stage1_select(preferences, history, history_ids, candidates, client)

    # Safety guard: if LLM hallucinated an id not in candidates, take top candidate
    if tmdb_id not in valid_ids:
        print(f"[warn] LLM returned id {tmdb_id} not in candidates — using top candidate")
        tmdb_id = int(candidates.iloc[0]["tmdb_id"])
        reasoning = "Selected as best semantic match."

    # 4. Stage 2 — write the description
    description = stage2_describe(preferences, tmdb_id, reasoning, mood, candidates, client)

    return {"tmdb_id": tmdb_id, "description": description}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie recommendation agent")
    parser.add_argument("--preferences", type=str)
    parser.add_argument("--history", type=str, help='Comma-separated titles, e.g. "Up, Coco"')
    args = parser.parse_args()

    preferences = (
        args.preferences.strip() if args.preferences and args.preferences.strip()
        else input("Preferences: ").strip()
    )
    history_raw = (
        args.history.strip() if args.history and args.history.strip()
        else input("Watch history (optional): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print(f"\n[mood detected: {detect_mood(preferences)}]")
    print("Thinking...\n")
    t0 = time.perf_counter()
    result = get_recommendation(preferences, history)
    elapsed = time.perf_counter() - t0

    movie = DF[DF["tmdb_id"] == result["tmdb_id"]]
    title = movie.iloc[0]["title"] if not movie.empty else "unknown"
    print(f'Recommended: "{title}" (tmdb_id={result["tmdb_id"]})')
    print(f'Description: {result["description"]}')
    print(f'\nServed in {elapsed:.2f}s')