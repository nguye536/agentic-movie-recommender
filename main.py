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
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import get_recommendation

app = FastAPI()


class HistoryItem(BaseModel):
    tmdb_id: int
    name: str


class RecommendRequest(BaseModel):
    user_id: int = 0
    preferences: str
    history: list[HistoryItem] = []


@app.get("/")
def health():
    return {"status": "ok"}


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
