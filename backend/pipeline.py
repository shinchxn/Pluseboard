"""
pipeline.py — Core generation pipeline for PulseBoard.

Flow:
  1. Call NVIDIA NIM (script model) with a structured JSON prompt
  2. Parse + validate response against LLMExplainerOutput schema
  3. Retry up to MAX_RETRIES on JSON validation failure
  4. Call NVIDIA NIM (html model) with the script to generate an animated HTML page
  5. Store the explainer JSON + animated HTML + provenance manifest to Backblaze B2 via genblaze-s3
  6. Return a populated Explainer dataclass to the API layer
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
# Script model: reasoning-heavy, used for structured JSON script generation
SCRIPT_MODEL = os.getenv("PULSEBOARD_SCRIPT_MODEL", os.getenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"))
# HTML model: instruction-following, used for animated HTML page generation
HTML_MODEL = os.getenv("PULSEBOARD_HTML_MODEL", os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct"))
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

HTML_SYSTEM_PROMPT = """\
You are a front-end engineer building a single, self-contained animated
HTML page for PulseBoard — a classroom smartboard tool that replaces static
PPT/PDF slides with a live, tap-to-advance visual walkthrough of a process.

You will be given a topic and a step-by-step script (title, phase labels,
descriptions). Your job is NOT to lay that content out as text cards or
bullet slides. Your job is to design a small animated "stage" that shows
the process actually happening, with text used only as a short caption
underneath — never as the primary content.

## The pattern to follow

Every PulseBoard explainer is built from the same five pieces. Adapt what
each piece represents to the specific topic — the pieces stay, the content
changes:

1. PARTICIPANTS — the fixed entities in the process, drawn as labeled boxes
   positioned spatially (e.g. two hosts on a network; a caller and a
   function on a call stack; the low/mid/high pointers in a binary search;
   a producer and a queue). Each participant box shows its own CURRENT
   STATE as a short live-updating label (e.g. "CLOSED" → "SYN_SENT" →
   "ESTABLISHED"; or "unvisited" → "visiting" → "done"). This state text
   must update via JS as the user steps through — never bake a single
   static state into the HTML.

2. THE MOVING ELEMENT — one visual thing that animates between or across
   participants on every step that involves a transfer, comparison, or
   state change (a packet flying along a wire; a token moving down a call
   stack; a pointer sliding across an array; a value being compared and
   highlighted). Position it with CSS transitions on `left`/`top`/`transform`
   (1s+ duration, easing), not an instant jump. It should carry a short
   label showing the concrete value involved at that step (e.g. "SYN
   seq=100", not just "SYN").

3. LIVE FIELDS — any numeric or state value that changes over the course of
   the process must be rendered as text that visibly updates via JS
   (`element.textContent = ...`) on each step — never as a static number
   baked into the HTML, and never described only in prose.

4. STEP CAPTION — a small caption area below the stage: a color-coded tag
   naming the current step, one to two sentences of caption text, and (if
   relevant) a monospace line of the exact field values active this step.
   This is the ONLY place prose text appears. Cap it at two sentences.

5. CONTROLS — Play/Pause, Next step, Reset, and a row of progress dots (one
   per step, filled as completed, enlarged on current). Support the
   spacebar as a keyboard shortcut for "next step." Buttons minimum 44px
   tall for touch/smartboard use.

## Hard rules

- Output ONE complete HTML document: inline `<style>` and inline `<script>`
  only. No external JS libraries, no build step, no network calls except
  optionally a webfont `<link>` — and if you use one, always include a
  system-font fallback in the `font-family` stack so the page works offline.
- Steps must be data: define a single JS array of step objects (state per
  participant, the moving element's from/to/label/color, the caption text,
  the live field values) and one `render(stepIndex)` function that reads
  from it. Do not hand-write repeated per-step markup blocks.
- Every step's animation must be replayable — clicking back to an earlier
  step re-triggers its motion, not just its end state.
- Use one consistent accent color per step/phase, applied identically to
  that step's moving element, its progress dot, and its caption tag.
- High-contrast, legible-from-across-a-room. Dark background with
  saturated accent colors. Avoid pale, low-contrast palettes.
- If the topic has no natural two-party exchange (e.g. an algorithm on one
  data structure), the "participants" become the structural elements
  (array cells, stack frames, tree nodes) and the "moving element" becomes
  a pointer, cursor, or highlighted comparison — the five pieces still apply.

## What NOT to produce

- A list of steps rendered as stacked text cards with headings and paragraphs.
- Any step whose only content is prose describing what changed.
- Numbers or state values baked as static HTML text rather than updated by JS.

Respond with ONLY the raw HTML document, starting with <!DOCTYPE html>.
No markdown fences, no commentary before or after.
"""


def _build_html_user_prompt(topic: str, script_json: str) -> str:
    return f"""Topic: {topic}

Step-by-step script (JSON):
{script_json}

Build the animated explainer page for this topic following the system
instructions exactly. Decide what the PARTICIPANTS and MOVING ELEMENT
represent for this specific topic before writing any code."""


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
def _get_nim_client():
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
        
        # Extract region from endpoint URL (e.g. https://s3.us-west-004.backblazeb2.com -> us-west-004)
        region_str = B2_ENDPOINT_URL.replace("https://s3.", "").replace(".backblazeb2.com", "") if B2_ENDPOINT_URL else "us-west-004"
        
        return S3StorageBackend.for_backblaze(
            bucket=B2_BUCKET_NAME,
            key_id=B2_KEY_ID,
            app_key=B2_APP_KEY,
            region=region_str,
            preflight=False,
        )
    except ImportError:
        logger.warning("genblaze-s3 not installed — B2 storage disabled.")
        return None


def _upload_to_b2(
    storage,
    run_id: str,
    topic_slug: str,
    payload: dict[str, Any],
    html_content: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Upload the explainer JSON, animated HTML, and a provenance manifest to B2.
    Returns (explainer_url, manifest_url, html_url) — all may be None on failure.
    """
    if storage is None:
        return None, None, None

    explainer_key = f"explainers/{topic_slug}/{run_id}.json"
    manifest_key = f"manifests/{run_id}_manifest.json"
    html_key = f"explainers/{topic_slug}/{run_id}.html"

    manifest_payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "provider": "nvidia-nim",
        "script_model": SCRIPT_MODEL,
        "html_model": HTML_MODEL,
        "topic": payload.get("topic", ""),
        "generated_at": payload.get("generated_at", ""),
        "explainer_key": explainer_key,
        "html_key": html_key if html_content else None,
    }

    base = B2_ENDPOINT_URL.rstrip("/")
    explainer_url = f"{base}/{B2_BUCKET_NAME}/{explainer_key}"
    manifest_url = f"{base}/{B2_BUCKET_NAME}/{manifest_key}"
    html_url = f"{base}/{B2_BUCKET_NAME}/{html_key}" if html_content else None
    payload["b2_url"] = explainer_url
    payload["manifest_url"] = manifest_url
    payload["html_url"] = html_url

    try:
        # Upload explainer JSON
        storage.put(
            key=explainer_key,
            data=json.dumps(payload, default=str).encode(),
            content_type="application/json",
        )

        # Upload animated HTML page
        if html_content:
            storage.put(
                key=html_key,
                data=html_content.encode("utf-8"),
                content_type="text/html; charset=utf-8",
            )
            logger.info("Uploaded animated HTML to B2: %s", html_url)

        # Upload manifest JSON
        storage.put(
            key=manifest_key,
            data=json.dumps(manifest_payload).encode(),
            content_type="application/json",
        )

        logger.info("Uploaded explainer to B2: %s", explainer_url)
        return explainer_url, manifest_url, html_url

    except Exception as exc:  # noqa: BLE001
        logger.error("B2 upload failed: %s", exc)
        return None, None, None


def _list_b2_explainers() -> list[dict[str, Any]]:
    """
    List all explainer objects from B2 under the 'explainers/' prefix.
    Returns a list of metadata dicts suitable for LibraryItem construction.
    """
    storage = _get_storage_backend()
    if storage is None:
        return []

    try:
        base = B2_ENDPOINT_URL.rstrip("/")
        items: list[dict[str, Any]] = []

        for key in storage.list(prefix="explainers/"):
            if not key.endswith(".json"):
                continue

            try:
                raw = storage.get(key)
                data: dict = json.loads(raw)
                # Key format: explainers/{slug}/{run_id}.json
                parts = key.split("/")
                run_id = parts[-1].replace(".json", "") if len(parts) >= 3 else key
                
                items.append({
                    "id": run_id,
                    "topic": data.get("topic", ""),
                    "title": data.get("title", ""),
                    "category": data.get("category", "general"),
                    "summary": data.get("summary", ""),
                    "steps_count": len(data.get("steps", [])),
                    "generated_at": data.get("generated_at", ""),
                    "b2_url": data.get("b2_url", f"{base}/{B2_BUCKET_NAME}/{key}"),
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
    storage = _get_storage_backend()
    if storage is None:
        return None

    try:
        target_key: str | None = None
        for key in storage.list(prefix="explainers/"):
            if explainer_id in key:
                target_key = key
                break
                
        if not target_key:
            return None

        raw = storage.get(target_key)
        return json.loads(raw)

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


# ── HTML generation ────────────────────────────────────────────────────────────
def _generate_html(client, topic: str, script: LLMExplainerOutput) -> str | None:
    """
    Second LLM pass: given the validated script, ask the HTML model to produce
    a single self-contained animated HTML explainer page.
    Returns the raw HTML string, or None on failure.
    """
    script_json = json.dumps(
        {
            "title": script.title,
            "category": script.category,
            "summary": script.summary,
            "steps": [s.model_dump() for s in script.steps],
        },
        indent=2,
    )
    try:
        logger.info("Generating animated HTML for topic=%r using model=%s", topic, HTML_MODEL)
        response = client.chat.completions.create(
            model=HTML_MODEL,
            messages=[
                {"role": "system", "content": HTML_SYSTEM_PROMPT},
                {"role": "user", "content": _build_html_user_prompt(topic, script_json)},
            ],
            temperature=0.4,
            max_tokens=8192,
        )
        html = response.choices[0].message.content or ""
        # Strip any accidental markdown code fences the model wraps around the HTML
        html = html.strip()
        for fence in ("```html", "```"):
            if html.startswith(fence):
                html = html[len(fence):].lstrip()
        if html.endswith("```"):
            html = html[:-3].rstrip()
        if not html.lstrip().startswith("<!DOCTYPE"):
            logger.warning("HTML model did not return a DOCTYPE document; discarding.")
            return None
        logger.info("Animated HTML generated successfully (%d bytes)", len(html))
        return html
    except Exception as exc:  # noqa: BLE001
        logger.error("HTML generation failed: %s", exc)
        return None


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
                model=SCRIPT_MODEL,
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
                "script_model": SCRIPT_MODEL,
                "html_model": HTML_MODEL,
            }

            # Generate animated HTML (second LLM pass)
            html_content = _generate_html(client, topic, llm_output)

            # Persist JSON + HTML + manifest to B2
            topic_slug = _slugify(topic)
            b2_url, manifest_url, html_url = _upload_to_b2(storage, run_id, topic_slug, payload, html_content)

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
                html_url=html_url,
                generated_at=generated_at,
            )

        except Exception as exc:  # noqa: BLE001
            # Don't retry configuration errors — they won't heal themselves
            if isinstance(exc, RuntimeError):
                raise
            last_error = exc
            logger.warning("Attempt %d failed: %s", attempt, exc)

    raise GenerationError(
        f"Failed to generate explainer for '{topic}' after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
