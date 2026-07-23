"""
main.py — FastAPI application for PulseBoard.

Endpoints:
  POST /api/generate           Generate a new explainer via NVIDIA NIM → store in B2
  GET  /api/library            List all saved explainers from B2
  GET  /api/explainer/{id}     Fetch a single explainer from B2 by run_id
  GET  /api/config-status      Debug: reports which integrations are configured
  GET  /health                 Health check
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

load_dotenv()

from pipeline import (  # noqa: E402
    NVIDIA_API_KEY,
    B2_KEY_ID,
    B2_APP_KEY,
    B2_BUCKET_NAME,
    B2_ENDPOINT_URL,
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

    # Startup configuration check
    missing: list[str] = []
    if not NVIDIA_API_KEY:
        missing.append("NVIDIA_API_KEY")
    if not B2_KEY_ID:
        missing.append("B2_KEY_ID")
    if not B2_APP_KEY:
        missing.append("B2_APP_KEY")
    if not B2_BUCKET_NAME:
        missing.append("B2_BUCKET_NAME")
    if not B2_ENDPOINT_URL:
        missing.append("B2_ENDPOINT_URL")

    if missing:
        logger.warning(
            "Missing environment variables — some features will be degraded: %s",
            ", ".join(missing),
        )
        if "NVIDIA_API_KEY" in missing:
            logger.error(
                "NVIDIA_API_KEY is required for generation. "
                "Set it in backend/.env before generating explainers."
            )
        b2_missing = [v for v in missing if v.startswith("B2_")]
        if b2_missing:
            logger.warning(
                "B2 storage is not configured (%s). "
                "Explainers will be generated but NOT persisted to cloud storage. "
                "html_url will be null until B2 credentials are set.",
                ", ".join(b2_missing),
            )
    else:
        logger.info("All required environment variables are set. Ready.")

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
    else [os.getenv("APP_URL", "")]
)

if not cors_origins[0]:
    logger.warning("APP_URL is not set — CORS will reject cross-origin requests in production.")

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


@app.get("/api/config-status", tags=["Meta"])
async def config_status():
    """
    Debug endpoint: reports which integrations are configured.
    Use this to diagnose why html_url may be null.
    """
    return {
        "nvidia_configured": bool(NVIDIA_API_KEY),
        "b2_configured": all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET_NAME, B2_ENDPOINT_URL]),
        "b2_key_id_set": bool(B2_KEY_ID),
        "b2_app_key_set": bool(B2_APP_KEY),
        "b2_bucket_set": bool(B2_BUCKET_NAME),
        "b2_endpoint_set": bool(B2_ENDPOINT_URL),
    }


@app.post(
    "/api/generate",
    response_model=GenerateResponse,
    status_code=201,
    tags=["Explainers"],
    summary="Generate a new visual explainer",
    responses={
        502: {"model": ErrorResponse, "description": "Generation failed after retries"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def generate(body: GenerateRequest) -> GenerateResponse:
    """
    Generate a structured visual explainer for the given topic.

    - Calls NVIDIA NIM (meta/llama-3.3-70b-instruct by default)
    - Validates the JSON output schema, retries up to MAX_RETRIES on failure
    - Grounds the script in Tavily-retrieved facts (if TAVILY_API_KEY is set)
    - Persists the explainer + provenance manifest to Backblaze B2
    - Returns the full explainer data
    """
    logger.info("Generate request: topic=%r", body.topic)

    try:
        explainer: Explainer = generate_explainer(body.topic)
    except GenerationError as exc:
        logger.error("GenerationError: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Generation service unavailable after retries — please try again.",
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
            html_url=data.get("html_url"),
            generated_at=datetime.fromisoformat(
                data.get("generated_at", datetime.utcnow().isoformat())
            ),
            # Part 1 — diagnostic fields (may not exist in older stored payloads)
            storage_configured=data.get("storage_configured", True),
            html_generation_failed=data.get("html_generation_failed", False),
            # Part 2 — research provenance (may not exist in older stored payloads)
            research_used=data.get("research_used", False),
            research_sources=data.get("research_sources", []),
            # Accuracy verification
            accuracy_verified=data.get("accuracy_verified", False),
        )
    except Exception as exc:
        logger.error("Failed to parse stored explainer %s: %s", explainer_id, exc)
        raise HTTPException(status_code=500, detail="Stored explainer data is malformed.") from exc

    return GenerateResponse(explainer=explainer)


@app.get(
    "/api/explainer/{explainer_id}/html",
    tags=["Explainers"],
    summary="Serve the animated HTML for an explainer (proxies from private B2 bucket)",
)
async def get_explainer_html(explainer_id: str):
    """
    Fetch the animated HTML file from Backblaze B2 using server-side credentials
    and stream it directly to the browser. This bypasses the private bucket
    restriction — the browser never needs direct B2 access.
    """
    # pyrefly: ignore [missing-import]
    from fastapi.responses import HTMLResponse
    from pipeline import _get_storage_backend

    storage = _get_storage_backend()
    if storage is None:
        raise HTTPException(status_code=503, detail="B2 storage is not configured.")

    try:
        # Find the HTML key for this run_id
        target_key: str | None = None
        page = storage.list(prefix="explainers/")
        for entry in page.entries:
            if explainer_id in entry.key and entry.key.endswith(".html"):
                target_key = entry.key
                break

        if not target_key:
            raise HTTPException(status_code=404, detail=f"HTML for explainer '{explainer_id}' not found.")

        raw = storage.get(target_key)
        html_content = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return HTMLResponse(content=html_content, status_code=200)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to proxy HTML for explainer %s: %s", explainer_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve HTML from storage.") from exc
