"""Data contracts shared by the agent, the API and the evaluation suite."""
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["urgent", "personal_question", "general_question", "statement"]


class Note(BaseModel):
    title: str
    text: str


class Chunk(BaseModel):
    id: str
    title: str
    text: str


class Hit(BaseModel):
    chunk: Chunk
    score: float


class HudPage(BaseModel):
    lines: list[str]
    seconds: float


class Suggestion(BaseModel):
    """A reply the user may choose to speak. Never spoken automatically."""

    text: str
    source_ids: list[str] = Field(default_factory=list)
    kind: Literal["grounded", "fallback", "social"] = "social"


class NodeTrace(BaseModel):
    node: str
    ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class TurnResult(BaseModel):
    turn_id: str
    intent: Intent
    caption: str
    hud: list[HudPage]
    suggestions: list[Suggestion]
    alert: str | None = None
    trace: list[NodeTrace]
    total_ms: float
    cost_usd: float
    brain: str
