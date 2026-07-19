"""
schemas.py — Pydantic models for PulseBoard API request/response contracts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Inbound ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="The technical / academic topic to generate an explainer for.",
        examples=["TCP Three-Way Handshake", "OSI Model Packet Flow"],
    )


# ── LLM output shape ─────────────────────────────────────────────────────────

class SlideStep(BaseModel):
    """One visual slide in the generated explainer."""
    phase: str = Field(..., description="Short phase title (e.g. 'Phase 1: SYN')")
    description: str = Field(..., description="1–3 sentence narrative for this step.")
    bullets: list[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="Concise technical bullet points (2–5 items).",
    )


class LLMExplainerOutput(BaseModel):
    """Schema enforced on raw LLM JSON output."""
    title: str = Field(..., description="Full display title of the explainer.")
    category: Literal["networking", "database", "operating-systems", "algorithms", "general"]
    summary: str = Field(..., description="One-sentence summary for library cards.")
    steps: list[SlideStep] = Field(
        ...,
        min_length=3,
        max_length=8,
        description="Ordered list of visual explainer slides.",
    )


# ── API outbound ──────────────────────────────────────────────────────────────

class Explainer(BaseModel):
    """Full explainer returned after successful generation."""
    id: str = Field(..., description="Unique run ID (UUID4).")
    topic: str
    title: str
    category: str
    summary: str
    steps: list[SlideStep]
    steps_count: int
    b2_url: str | None = Field(None, description="Durable B2 URL for the stored JSON.")
    manifest_url: str | None = Field(None, description="Provenance manifest B2 URL.")
    html_url: str | None = Field(None, description="B2 URL for the animated HTML page.")
    generated_at: datetime
    # Part 1 — diagnostic fields
    storage_configured: bool = Field(False, description="True if B2 was configured at generation time.")
    html_generation_failed: bool = Field(False, description="True if the HTML model failed both attempts.")
    # Part 2 — Tavily research provenance
    research_used: bool = Field(False, description="True if Tavily grounding was applied.")
    research_sources: list[str] = Field(default_factory=list, description="Source URLs from Tavily.")


class LibraryItem(BaseModel):
    """Compact metadata card for the library listing."""
    id: str
    topic: str
    title: str
    category: str
    summary: str
    steps_count: int
    generated_at: datetime
    b2_url: str


class GenerateResponse(BaseModel):
    success: bool = True
    explainer: Explainer


class LibraryResponse(BaseModel):
    success: bool = True
    items: list[LibraryItem]
    total: int


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str | None = None
