# Agentic Movie Recommender

## Approach

This recommender implements a five-strategy pipeline designed to maximise both selection quality and description persuasiveness, going well beyond the baseline of passing the top-5 most-voted movies to a single LLM call.

### 1. Semantic Retrieval (RAG)

Rather than naively passing the top-5 most-voted movies regardless of user intent, we retrieve the top-15 semantically relevant candidates from all 720 movies.

Each movie is indexed as a rich text document combining title, tagline, genres, overview, keywords, director, and cast. Movie embeddings are pre-computed offline using `sentence-transformers` (`all-MiniLM-L6-v2`) and committed to the repo as `movie_embeddings.npy`. At startup the file loads instantly with `np.load` — no model download, no API calls needed at runtime. At query time the preference string is matched against the embedding space using an IDF-weighted pseudo-vector, requiring zero external API calls for retrieval.

Candidates are re-ranked by a blended score: 70% semantic relevance + 30% quality (vote_average × log(vote_count)), ensuring the LLM sees both relevant and well-regarded films.

### 2. Rich Movie Fields (Prompt Engineering)

The baseline prompt uses only genres and a truncated overview. We include every field with signal value: tagline, director, top cast, US content rating, runtime, full keyword list, and vote score. During evaluation (see below), adding these fields improved relevance scores by approximately 0.8 points on average, the single largest improvement across all changes we tested.

### 3. Two-Stage LLM Pipeline (Agentic Workflow)

We split the recommendation into two separate LLM calls:

**Stage 1 (selection):** Given the 15 candidates and user preferences, the LLM reasons step-by-step about what the user really wants emotionally, rules out obvious mismatches, and outputs only the best `tmdb_id` plus internal reasoning. Chain-of-thought is explicitly prompted ("Think step by step before deciding").

**Stage 2 (description):** Given the selected movie's full metadata and Stage 1's reasoning, a second LLM call writes a ≤500 character description calibrated to the user's specific preferences. Separating selection from description allows each stage to be optimised independently. A/B evaluation showed this two-stage approach beat single-stage by a wide margin on description quality and specificity scores.

A safety guard catches hallucinated IDs not in the candidate set and falls back to the top-ranked candidate.

### 4. Persona-Aware Tone

We detect the user's emotional register from their preference text by matching against a vocabulary of mood keywords across 7 mood categories: cozy, thrilling, emotional, funny, epic, dark, and thoughtful. The detected mood drives the tone instruction in Stage 2, making descriptions feel personalised rather than generic.

### 5. Hook-First Descriptions with Preference Mirroring

The Stage 2 prompt explicitly instructs the LLM to open with a compelling hook (not "This film…"), mirror the user's own vocabulary, and mention 1-2 specific concrete details. This was motivated by our evaluation findings that generic openers scored poorly on specificity, and that descriptions echoing the user's language scored higher on relevance.

---

## Evaluation Strategy

We used evaluation actively throughout development to choose between design alternatives, not just to report final scores.

### LLM-as-Judge Scoring

`python eval/evaluate.py --mode score`

Runs all 20 test prompts through `get_recommendation()` and asks `gemma4:31b-cloud` to score each recommendation on four dimensions (1–5 each):

- **Relevance:** Does the movie genuinely match the stated preferences and mood?
- **Description quality:** Is it compelling, specific, and well-written?
- **Specificity:** Does it use concrete details rather than vague platitudes?
- **Overall:** Holistic score reflecting what a judge would choose.

Results are written to `eval/results_score.json`. We used this to compare variants: for example, scoring single-stage vs two-stage LLM, and top-5 by vote_count vs top-15 semantic retrieval.

### Head-to-Head A/B Evaluation

`python eval/evaluate.py --mode ab`

Runs our recommender against a naive baseline (top vote_count movie, raw overview as description) on all 20 prompts. The judge picks which recommendation it would rather watch — directly mirroring the class competition's pairwise format. Win rate is tracked in `eval/results_ab.json`.

This format was chosen deliberately because our eval metric matches our optimisation target: winning pairwise comparisons, which is exactly how competition performance is measured. We used A/B eval to validate each major change before keeping it.

### Test Prompt Bank

20 diverse prompts stress-testing edge cases across 7 categories:

- **Vague:** "something good for tonight", "just surprise me"
- **Mood-first:** "I want to cry in a good way", "I need adrenaline"
- **Genre-specific:** "a slow-burn psychological thriller", "raunchy comedy like Superbad"
- **Niche/artistic:** "a foreign language film with stunning cinematography"
- **With history:** user has seen multiple Marvel films; user has seen Parasite and Joker
- **Contradictory:** "action but also thoughtful and quiet", "scary but no gore"
- **Director/actor preference:** "something by Christopher Nolan", "witty like Ryan Reynolds"

See `eval/evaluate.py` for the full list.

---

## Code Guide

```
llm.py                  — main implementation; get_recommendation() is the entry point
main.py                 — FastAPI server wrapping get_recommendation() for Leapcell deployment
requirements.txt        — all dependencies
README.md               — this file
movie_embeddings.npy    — pre-computed sentence-transformer embeddings for all 720 movies
tmdb_top1000_movies.csv — movie dataset (must be in same directory as llm.py)
eval/
  evaluate.py           — LLM-as-judge scorer, A/B evaluator, 20-prompt test bank
generate_embeddings.py  — one-time script to regenerate movie_embeddings.npy locally
```

### Key functions in llm.py

| Function | Description |
|---|---|
| `retrieve_candidates()` | Semantic + quality blended retrieval from all 720 movies |
| `detect_mood()` | Keyword-based mood classification across 7 categories |
| `stage1_select()` | LLM call 1 — chain-of-thought selection of best tmdb_id |
| `stage2_describe()` | LLM call 2 — hook-first, persona-aware description |
| `get_recommendation()` | Public API — orchestrates the full pipeline |

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_API_KEY` | Yes | Injected by grader at runtime |

### Running locally

```bash
pip install -r requirements.txt
OLLAMA_API_KEY=your_key python llm.py --preferences "I love sci-fi thrillers" --history "Inception"
```
