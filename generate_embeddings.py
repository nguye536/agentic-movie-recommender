"""
Run this ONCE locally to pre-compute movie embeddings.
Commit the output file (movie_embeddings.npy) to your GitHub repo.
Leapcell loads it instantly at startup — no model download needed there.

Usage:
  pip install sentence-transformers
  python generate_embeddings.py
"""

import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# Load dataset
DATA_PATH = "tmdb_top1000_movies.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "tmdb_top1000_movies.xlsx"

df = pd.read_excel(DATA_PATH) if DATA_PATH.endswith(".xlsx") else \
     pd.read_csv(DATA_PATH, encoding="utf-8", encoding_errors="replace")

def _safe(val, maxlen=None):
    if pd.isna(val): return ""
    s = str(val)
    return s[:maxlen] if maxlen else s

def build_doc(row):
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

docs = [build_doc(row) for _, row in df.iterrows()]
print(f"Embedding {len(docs)} movies with all-MiniLM-L6-v2...")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
embeddings = embeddings.astype(np.float32)

np.save("movie_embeddings.npy", embeddings)
print(f"\nSaved movie_embeddings.npy — shape: {embeddings.shape}")
print("Now run:")
print("  git add movie_embeddings.npy")
print("  git commit -m 'add precomputed movie embeddings'")
print("  git push")
