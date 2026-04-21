"""
Run this ONCE locally to pre-compute movie embeddings using Ollama Cloud.
Commit the output file (movie_embeddings.npy) to your GitHub repo.
Leapcell will load it instantly — no API calls needed at startup.

Usage:
  OLLAMA_API_KEY=your_key python generate_embeddings.py
"""

import os
import numpy as np
import pandas as pd
import ollama

EMBED_MODEL = "nomic-embed-text"   # fast, high-quality, available on Ollama Cloud

client = ollama.Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
)

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
print(f"Embedding {len(docs)} movies with {EMBED_MODEL}...")

# Embed in batches of 32
all_embeddings = []
batch_size = 32
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    resp = client.embed(model=EMBED_MODEL, input=batch)
    all_embeddings.extend(resp["embeddings"])
    print(f"  {min(i+batch_size, len(docs))}/{len(docs)} done")

embeddings = np.array(all_embeddings, dtype=np.float32)

# Normalise to unit length so dot product = cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / (norms + 1e-9)

np.save("movie_embeddings.npy", embeddings)
print(f"\nSaved movie_embeddings.npy — shape: {embeddings.shape}")
print("Now commit this file to your GitHub repo!")
