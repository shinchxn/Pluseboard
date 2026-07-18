"""
main.py — FastAPI application for PulseBoard.

Endpoints:
  POST /api/generate           Generate a new explainer via NVIDIA NIM → store in B2
  GET  /api/library            List all saved explainers from B2
  GET  /api/explainer/{id}     Fetch a single explainer from B2 by run_id
  GET  /health                 Health check
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from pipeline import (  # noqa: E402
    GenerationError,
    _fetch_b2_explainer,
    _list_b2_explainers,
    generate_explainer,
)
from schemas import (  # noqa: E402
    ErrorResponse,
    Explainer,
    GenerateRequest,
    GenerateResponse,
    LibraryItem,
    LibraryResponse,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")


# ── App lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("PulseBoard API starting — environment=%s", ENVIRONMENT)
    yield
    logger.info("PulseBoard API shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PulseBoard API",
    description=(
        "Backend for PulseBoard — generates visual explainers via NVIDIA NIM "
        "and stores them in Backblaze B2 with provenance manifests."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in dev so file:// HTML pages can call the API
cors_origins = (
    ["*"]
    if ENVIRONMENT == "development"
    else [os.getenv("APP_URL", "http://localhost:8000")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Global error handler ───────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if ENVIRONMENT == "development" else None,
        ).model_dump(),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
async def health_check():
    """Simple health check — returns OK if the server is running."""
    return {"status": "ok", "service": "PulseBoard API"}


@app.post(
    "/api/generate",
    response_model=GenerateResponse,
    status_code=201,
    tags=["Explainers"],
    summary="Generate a new visual explainer",
    responses={
        422: {"model": ErrorResponse, "description": "Generation failed after retries"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def generate(body: GenerateRequest) -> GenerateResponse:
    """
    Generate a structured visual explainer for the given topic.

    - Calls NVIDIA NIM (meta/llama-3.3-70b-instruct by default)
    - Validates the JSON output schema, retries up to MAX_RETRIES on failure
    - Persists the explainer + provenance manifest to Backblaze B2
    - Returns the full explainer data
    """
    logger.info("Generate request: topic=%r", body.topic)

    try:
        explainer: Explainer = generate_explainer(body.topic)
    except GenerationError as exc:
        logger.error("GenerationError: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return GenerateResponse(explainer=explainer)


@app.get(
    "/api/library",
    response_model=LibraryResponse,
    tags=["Library"],
    summary="List all saved explainers",
)
async def library() -> LibraryResponse:
    """
    Return all explainers stored in the Backblaze B2 bucket,
    sorted by generation date (newest first).
    """
    raw_items = _list_b2_explainers()

    items: list[LibraryItem] = []
    for raw in raw_items:
        try:
            items.append(LibraryItem(**raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping malformed library item: %s — %s", raw, exc)

    return LibraryResponse(items=items, total=len(items))


@app.get(
    "/api/explainer/{explainer_id}",
    response_model=GenerateResponse,
    tags=["Explainers"],
    summary="Fetch a single saved explainer",
    responses={
        404: {"model": ErrorResponse, "description": "Explainer not found in B2"},
    },
)
async def get_explainer(explainer_id: str) -> GenerateResponse:
    """
    Retrieve a previously generated explainer from Backblaze B2 by its run_id.
    """
    data = _fetch_b2_explainer(explainer_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Explainer '{explainer_id}' not found in library.",
        )

    # Re-hydrate into Explainer model
    try:
        from datetime import datetime
        steps_raw = data.get("steps", [])
        from schemas import SlideStep
        steps = [SlideStep(**s) for s in steps_raw]

        explainer = Explainer(
            id=data.get("id", explainer_id),
            topic=data.get("topic", ""),
            title=data.get("title", ""),
            category=data.get("category", "general"),
            summary=data.get("summary", ""),
            steps=steps,
            steps_count=len(steps),
            b2_url=data.get("b2_url"),
            manifest_url=data.get("manifest_url"),
            generated_at=datetime.fromisoformat(
                data.get("generated_at", datetime.utcnow().isoformat())
            ),
        )
    except Exception as exc:
        logger.error("Failed to parse stored explainer %s: %s", explainer_id, exc)
        raise HTTPException(status_code=500, detail="Stored explainer data is malformed.") from exc

    return GenerateResponse(explainer=explainer)
