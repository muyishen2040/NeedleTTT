"""Local-only API using fictional items and an UNTRAINED teaching model.

Run with:
    uvicorn apps.api.main:app --host 127.0.0.1 --port 8000

No cloud hosting, real data, persistence, or trained checkpoint is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from needlettt.contracts import ItemCatalog
from needlettt.models.interfaces import ModelState
from needlettt.models.reference_numpy import TeachingRecommender

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ItemCatalog.from_json(ROOT / "data/examples/catalog.json")
VARIANTS = ("base", "vector", "matrix")
MODELS = {variant: TeachingRecommender(CATALOG, variant=variant) for variant in VARIANTS}
MODEL_STATUS = "toy_untrained"


class RatingInput(BaseModel):
    item_id: str
    rating: int = Field(ge=1, le=5, strict=True)


class CreateSessionInput(BaseModel):
    variant: Literal["base", "vector", "matrix"] = "matrix"


class ReplayInput(BaseModel):
    events: list[RatingInput]


@dataclass
class SessionRecord:
    preferred_variant: str = "matrix"
    history: list[RatingInput] = field(default_factory=list)
    states: dict[str, ModelState] = field(
        default_factory=lambda: {
            variant: model.init_state() for variant, model in MODELS.items()
        }
    )


SESSIONS: dict[str, SessionRecord] = {}

app = FastAPI(title="NeedleTTT local teaching API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_session(session_id: str) -> SessionRecord:
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown session")
    return SESSIONS[session_id]


def valid_item_idx(item_id: str) -> int:
    item = CATALOG.by_id.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown item_id: {item_id}")
    return item.item_idx


def apply_rating(session: SessionRecord, event: RatingInput) -> None:
    idx = valid_item_idx(event.item_id)
    for variant, model in MODELS.items():
        session.states[variant] = model.update(
            session.states[variant], item_ids=[idx], ratings=[event.rating]
        )
    session.history.append(event)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_status": MODEL_STATUS, "variants": VARIANTS}


@app.get("/items")
def items(query: str = "", limit: int = Query(50, ge=1, le=500)) -> dict[str, object]:
    return {"items": CATALOG.search(query=query, limit=limit), "model_status": MODEL_STATUS}


@app.post("/sessions")
def create_session(payload: CreateSessionInput | None = None) -> dict[str, object]:
    session_id = str(uuid4())
    preferred = payload.variant if payload is not None else "matrix"
    SESSIONS[session_id] = SessionRecord(preferred_variant=preferred)
    return {
        "session_id": session_id, "model_status": MODEL_STATUS,
        "variants": VARIANTS, "default_variant": preferred,
    }


@app.get("/sessions/{session_id}/recommendations")
def recommendations(
    session_id: str,
    variant: Literal["base", "vector", "matrix"] | None = None,
    limit: int = Query(10, ge=1, le=100),
) -> dict[str, object]:
    session = get_session(session_id)
    selected = variant or session.preferred_variant
    seen_ids = {rating.item_id for rating in session.history}
    candidates = [item for item in CATALOG.items if item.item_id not in seen_ids]
    if not candidates:
        ranked = []
    else:
        idx = [item.item_idx for item in candidates]
        raw_scores = MODELS[selected].recommend(session.states[selected], [idx])[0]
        sorted_positions = sorted(
            range(len(candidates)), key=lambda j: (-float(raw_scores[j]), idx[j])
        )
        ranked = [
            {
                "item_id": candidates[j].item_id,
                "title": candidates[j].title,
                "categories": list(candidates[j].categories),
                "image_url": candidates[j].image_url,
                "score": float(raw_scores[j]),
                "rank": rank,
            }
            for rank, j in enumerate(sorted_positions[:limit], start=1)
        ]
    return {
        "session_id": session_id,
        "variant": selected,
        "model_status": MODEL_STATUS,
        "history_length": len(session.history),
        "recommendations": ranked,
    }


@app.post("/sessions/{session_id}/ratings")
def rate(session_id: str, event: RatingInput) -> dict[str, object]:
    session = get_session(session_id)
    apply_rating(session, event)
    return {"history_length": len(session.history), "model_status": MODEL_STATUS}


@app.post("/sessions/{session_id}/reset")
def reset(session_id: str) -> dict[str, object]:
    session = get_session(session_id)
    session.history.clear()
    session.states = {name: model.init_state() for name, model in MODELS.items()}
    return {"history_length": 0, "model_status": MODEL_STATUS}


@app.post("/sessions/{session_id}/replay")
def replay(session_id: str, payload: ReplayInput) -> dict[str, object]:
    session = get_session(session_id)
    # Validate the whole replay before discarding the existing state.
    for event in payload.events:
        valid_item_idx(event.item_id)
    reset(session_id)
    for event in payload.events:
        apply_rating(session, event)
    return {"history_length": len(session.history), "model_status": MODEL_STATUS}
