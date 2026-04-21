"""
main.py — Leapcell entry point.
"""

import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from llm import get_recommendation, fuzzy_search_titles, DF

app = FastAPI()
HTML_PATH = Path(__file__).parent / "index.html"


class HistoryItem(BaseModel):
    tmdb_id: int
    name: str


class RecommendRequest(BaseModel):
    user_id: int = 0
    preferences: str
    history: list[HistoryItem] = []


@app.get("/", response_class=HTMLResponse)
def index():
    if HTML_PATH.exists():
        return HTML_PATH.read_text()
    return HTMLResponse("<h1>CineMatch</h1>")


@app.get("/search")
def search_movies(q: str, limit: int = 5):
    return {"results": fuzzy_search_titles(q, limit)}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        history_names = [h.name for h in req.history]
        history_ids = [h.tmdb_id for h in req.history]
        result = get_recommendation(req.preferences, history_names, history_ids)

        # Look up title and poster from dataset
        movie = DF[DF["tmdb_id"] == result["tmdb_id"]]
        title = str(movie.iloc[0]["title"]) if not movie.empty else "Unknown"
        year = int(movie.iloc[0]["year"]) if not movie.empty and pd.notna(movie.iloc[0]["year"]) else ""
        poster = str(movie.iloc[0]["poster_path"]) if not movie.empty and pd.notna(movie.iloc[0]["poster_path"]) else ""
        genres = str(movie.iloc[0]["genres"]) if not movie.empty and pd.notna(movie.iloc[0]["genres"]) else ""

        return {
            "tmdb_id": result["tmdb_id"],
            "user_id": req.user_id,
            "title": title,
            "year": year,
            "poster": poster,
            "genres": genres,
            "description": result["description"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
