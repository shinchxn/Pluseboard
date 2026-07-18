"""
pipeline.py — Core generation pipeline for PulseBoard.

Flow:
  1. Call NVIDIA NIM (via OpenAI-compat client) with a structured JSON prompt
  2. Parse + validate response against LLMExplainerOutput schema
  3. Retry up to MAX_RETRIES on validation failure
  4. Store the explainer JSON + provenance manifest to Backblaze B2 via genblaze-s3
  5. Return a populated Explainer dataclass to the API layer
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from schemas import Explainer, LLMExplainerOutput, SlideStep

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
B2_KEY_ID = os.getenv("B2_KEY_ID", "")
B2_APP_KEY = os.getenv("B2_APP_KEY", "")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "pulseboard-explainers")
B2_ENDPOINT_URL = os.getenv("B2_ENDPOINT_URL", "")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TARGET_STEPS = int(os.getenv("TARGET_STEPS", "4"))


# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert academic curriculum designer and technical illustrator.
Your job is to produce concise, high-contrast visual explainer content for
classroom smartboards. All output must be valid, parseable JSON — nothing else.
"""

def _build_user_prompt(topic: str, target_steps: int = TARGET_STEPS) -> str:
    return f"""
Generate a structured visual explainer for the following academic/technical topic:

TOPIC: "{topic}"

Return ONLY a JSON object matching this exact schema (no markdown, no extra keys):

{{
  "title": "<Full display title of the explainer>",
  "category": "<one of: networking | database | operating-systems | algorithms | general>",
  "summary": "<one sentence summary for library cards>",
  "steps": [
    {{
      "phase": "<Short phase label, e.g. 'Phase 1: SYN'>",
      "description": "<1-3 sentence narrative explaining this step clearly for students>",
      "bullets": [
        "<concise technical bullet point>",
        "<concise technical bullet point>",
        "<concise technical bullet point>"
      ]
    }}
  ]
}}

Requirements:
- Produce exactly {target_steps} steps (minimum 3, maximum 8).
- Each step must have 2–5 bullets.
- Use precise, academic language suitable for computer science students.
- The flow must be sequential and causal — each step leads logically to the next.
- Return ONLY the JSON object, starting with {{ and ending with }}.
"""


# ── NIM client ────────────────────────────────────────────────────────────────
def _get_nim_client() -> OpenAI:
    """Return an OpenAI client pointed at NVIDIA NIM."""
    if not NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Add it to backend/.env"
        )
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
    )


# ── B2 storage (genblaze-s3) ──────────────────────────────────────────────────
def _get_storage_backend():
    """Return a genblaze-s3 S3StorageBackend for Backblaze B2, or None if unconfigured."""
    if not all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET_NAME, B2_ENDPOINT_URL]):
        logger.warning(
            "B2 credentials incomplete — explainers will NOT be persisted to B2. "
            "Set B2_KEY_ID, B2_APP_KEY, B2_BUCKET_NAME, B2_ENDPOINT_URL in .env."
        )
        return None
    try:
        from genblaze_s3 import S3StorageBackend  # type: ignore
        return S3StorageBackend(
            bucket_name=B2_BUCKET_NAME,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
            endpoint_url=B2_ENDPOINT_URL,
            region_name="us-west-004",  # B2 ignores this but boto3 requires it
        )
    except ImportError:
        logger.warning("genblaze-s3 not installed — B2 storage disabled.")
        return None


def _upload_to_b2(
    storage,
    run_id: str,
    topic_slug: str,
    payload: dict[str, Any],
) -> tuple[str | None, str | None]:
    """
    Upload the explainer JSON and a provenance manifest to B2.
    Returns (explainer_url, manifest_url) — both may be None on failure.
    """
    if storage is None:
        return None, None

    explainer_key = f"explainers/{topic_slug}/{run_id}.json"
    manifest_key = f"manifests/{run_id}_manifest.json"

    manifest_payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "provider": "nvidia-nim",
        "model": NVIDIA_MODEL,
        "topic": payload.get("topic", ""),
        "generated_at": payload.get("generated_at", ""),
        "explainer_key": explainer_key,
    }

    try:
        import boto3  # type: ignore
        s3 = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT_URL,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
        )

        # Upload explainer JSON
        s3.put_object(
            Bucket=B2_BUCKET_NAME,
            Key=explainer_key,
            Body=json.dumps(payload, default=str),
            ContentType="application/json",
        )

        # Upload manifest JSON
        s3.put_object(
            Bucket=B2_BUCKET_NAME,
            Key=manifest_key,
            Body=json.dumps(manifest_payload),
            ContentType="application/json",
        )

        # Build public-friendly URLs
        base = B2_ENDPOINT_URL.rstrip("/")
        explainer_url = f"{base}/{B2_BUCKET_NAME}/{explainer_key}"
        manifest_url = f"{base}/{B2_BUCKET_NAME}/{manifest_key}"

        logger.info("Uploaded explainer to B2: %s", explainer_url)
        return explainer_url, manifest_url

    except Exception as exc:  # noqa: BLE001
        logger.error("B2 upload failed: %s", exc)
        return None, None


def _list_b2_explainers() -> list[dict[str, Any]]:
    """
    List all explainer objects from B2 under the 'explainers/' prefix.
    Returns a list of metadata dicts suitable for LibraryItem construction.
    """
    if not all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET_NAME, B2_ENDPOINT_URL]):
        return []

    try:
        import boto3  # type: ignore
        s3 = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT_URL,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
        )

        paginator = s3.get_paginator("list_objects_v2")
        items: list[dict[str, Any]] = []
        base = B2_ENDPOINT_URL.rstrip("/")

        for page in paginator.paginate(Bucket=B2_BUCKET_NAME, Prefix="explainers/"):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if not key.endswith(".json"):
                    continue
                # Key format: explainers/{slug}/{run_id}.json
                parts = key.split("/")
                run_id = parts[-1].replace(".json", "") if len(parts) >= 3 else key
                b2_url = f"{base}/{B2_BUCKET_NAME}/{key}"

                # Fetch the object to read metadata
                try:
                    resp = s3.get_object(Bucket=B2_BUCKET_NAME, Key=key)
                    data: dict = json.loads(resp["Body"].read())
                    items.append({
                        "id": run_id,
                        "topic": data.get("topic", ""),
                        "title": data.get("title", ""),
                        "category": data.get("category", "general"),
                        "summary": data.get("summary", ""),
                        "steps_count": len(data.get("steps", [])),
                        "generated_at": data.get("generated_at", obj["LastModified"].isoformat()),
                        "b2_url": b2_url,
                    })
                except Exception as fetch_err:  # noqa: BLE001
                    logger.warning("Could not fetch explainer %s: %s", key, fetch_err)

        # Sort newest first
        items.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
        return items

    except Exception as exc:  # noqa: BLE001
        logger.error("B2 list failed: %s", exc)
        return []


def _fetch_b2_explainer(explainer_id: str) -> dict[str, Any] | None:
    """Fetch a single explainer JSON from B2 by run_id (searches all slugs)."""
    if not all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET_NAME, B2_ENDPOINT_URL]):
        return None

    try:
        import boto3  # type: ignore
        s3 = boto3.client(
            "s3",
            endpoint_url=B2_ENDPOINT_URL,
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY,
        )

        # List to find the key containing the run_id
        paginator = s3.get_paginator("list_objects_v2")
        target_key: str | None = None

        for page in paginator.paginate(Bucket=B2_BUCKET_NAME, Prefix="explainers/"):
            for obj in page.get("Contents", []):
                if explainer_id in obj["Key"]:
                    target_key = obj["Key"]
                    break
            if target_key:
                break

        if not target_key:
            return None

        resp = s3.get_object(Bucket=B2_BUCKET_NAME, Key=target_key)
        return json.loads(resp["Body"].read())

    except Exception as exc:  # noqa: BLE001
        logger.error("B2 fetch failed for id=%s: %s", explainer_id, exc)
        return None


# ── Slug helper ───────────────────────────────────────────────────────────────
def _slugify(text: str) -> str:
    """Convert a topic string to a safe B2/S3 key segment."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:60]


# ── Core generation with retry ────────────────────────────────────────────────
class GenerationError(Exception):
    """Raised when the pipeline fails after exhausting all retries."""


def generate_explainer(topic: str) -> Explainer:
    """
    Full generate → validate → retry → B2-store pipeline.
    Returns a populated Explainer on success, raises GenerationError on failure.
    """
    client = _get_nim_client()
    storage = _get_storage_backend()
    run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        # Slightly raise temperature on retries to escape stuck outputs
        temperature = 0.2 + (attempt - 1) * 0.15
        logger.info(
            "Generation attempt %d/%d for topic=%r (temp=%.2f)",
            attempt, MAX_RETRIES, topic, temperature,
        )

        try:
            response = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(topic)},
                ],
                temperature=temperature,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            raw_content = response.choices[0].message.content or ""
            logger.debug("Raw LLM response (attempt %d): %s", attempt, raw_content[:500])

            # Parse JSON
            try:
                raw_json: dict = json.loads(raw_content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

            # Validate against schema
            try:
                llm_output = LLMExplainerOutput.model_validate(raw_json)
            except ValidationError as exc:
                raise ValueError(f"LLM output failed schema validation: {exc}") from exc

            # Construct storage payload
            steps_dicts = [s.model_dump() for s in llm_output.steps]
            payload: dict[str, Any] = {
                "id": run_id,
                "topic": topic,
                "title": llm_output.title,
                "category": llm_output.category,
                "summary": llm_output.summary,
                "steps": steps_dicts,
                "generated_at": generated_at.isoformat(),
                "provider": "nvidia-nim",
                "model": NVIDIA_MODEL,
            }

            # Persist to B2
            topic_slug = _slugify(topic)
            b2_url, manifest_url = _upload_to_b2(storage, run_id, topic_slug, payload)

            return Explainer(
                id=run_id,
                topic=topic,
                title=llm_output.title,
                category=llm_output.category,
                summary=llm_output.summary,
                steps=llm_output.steps,
                steps_count=len(llm_output.steps),
                b2_url=b2_url,
                manifest_url=manifest_url,
                generated_at=generated_at,
            )

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)

    raise GenerationError(
        f"Failed to generate explainer for '{topic}' after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
