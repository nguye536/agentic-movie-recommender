"""
main.py — Leapcell entry point.

Grader calls POST /recommend with this exact body:
  {
    "user_id": 1,
    "preferences": "I love superheroes and feel-good buddy cop stories.",
    "history": [{"tmdb_id": 24428, "name": "The Avengers"}]
  }

Expected response:
  {
    "tmdb_id": 284053,
    "user_id": 1,
    "description": "..."
  }
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from llm import get_recommendation

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
    return HTMLResponse("<h1>CineMatch</h1><p>UI not found.</p>")


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        history_names = [h.name for h in req.history]
        history_ids = [h.tmdb_id for h in req.history]
        result = get_recommendation(req.preferences, history_names, history_ids)
        return {
            "tmdb_id": result["tmdb_id"],
            "user_id": req.user_id,
            "description": result["description"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
