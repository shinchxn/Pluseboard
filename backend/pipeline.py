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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Part 0 — Composition engine feature flag (default: off)
USE_COMPOSITION_ENGINE = os.getenv("ENABLE_COMPOSITION_ENGINE", "false").lower() == "true"


# ── Prompt templates ──────────────────────────────────────────────────────────
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

# HyperFrames composition contract addendum (appended when USE_COMPOSITION_ENGINE=True)
HYPERFRAMES_SYSTEM_PROMPT_ADDENDUM = """\

## HyperFrames Composition Structure

Structure the output as a HyperFrames composition: the root element needs
`data-composition-id="pulseboard-explainer"`, `data-width="1920"`,
`data-height="1080"`. Each step is a `<div class="clip" data-start="{N}"
data-duration="{D}">` positioned sequentially on the timeline. Use the
`packet-motion`, `phase-caption`, and `progress-dots` block patterns
provided below for each step's content — assemble from these rather than
inventing new animation code. Use `element.animate()` (WAAPI) for any
motion, never raw CSS transitions, so playback stays frame-accurate when
seeked externally.

### Block: progress-dots
```html
<div class="pb-progress-dots" data-step-count="4" data-current-step="0">
  <style>
    .pb-progress-dots { display: flex; gap: 8px; }
    .pb-progress-dots .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #3a3a4a;
    }
    .pb-progress-dots .dot.active { background: #7c6fff; }
  </style>
  <script>
    (function (root) {
      const count = parseInt(root.dataset.stepCount, 10);
      root.innerHTML += Array.from({ length: count })
        .map((_, i) => `<div class="dot" data-index="${i}"></div>`)
        .join("");
      root.setActiveStep = function (i) {
        root.querySelectorAll(".dot").forEach((d, idx) =>
          d.classList.toggle("active", idx === i)
        );
      };
      root.setActiveStep(parseInt(root.dataset.currentStep, 10) || 0);
    })(document.currentScript.closest(".pb-progress-dots"));
  </script>
</div>
```

### Block: packet-motion
```html
<div class="pb-packet-motion"
     data-from-x="100" data-from-y="200"
     data-to-x="800"   data-to-y="200"
     data-label="SYN seq=100"
     data-color="#7c6fff"
     data-duration-ms="1200">
  <style>
    .pb-packet-motion .packet {
      position: absolute;
      padding: 6px 14px;
      border-radius: 20px;
      font: 700 13px/1 monospace;
      color: #fff;
      pointer-events: none;
    }
  </style>
  <script>
    (function (root) {
      const p = document.createElement("div");
      p.className = "packet";
      p.textContent = root.dataset.label;
      p.style.background = root.dataset.color;
      const fx = parseFloat(root.dataset.fromX), fy = parseFloat(root.dataset.fromY);
      const tx = parseFloat(root.dataset.toX),   ty = parseFloat(root.dataset.toY);
      const dur = parseInt(root.dataset.durationMs, 10) || 1200;
      p.style.left = fx + "px"; p.style.top = fy + "px";
      root.appendChild(p);
      root.play = function () {
        p.animate(
          [{ transform: `translate(0,0)` }, { transform: `translate(${tx - fx}px,${ty - fy}px)` }],
          { duration: dur, easing: "cubic-bezier(.4,0,.2,1)", fill: "forwards" }
        );
      };
    })(document.currentScript.closest(".pb-packet-motion"));
  </script>
</div>
```

### Block: phase-caption
```html
<div class="pb-phase-caption"
     data-phase="Phase 1: SYN"
     data-color="#7c6fff"
     data-description="Client sends SYN to initiate connection."
     data-fields="seq=100 | ack=0 | flags=SYN">
  <style>
    .pb-phase-caption { padding: 16px 24px; border-radius: 12px; background: #13131a; }
    .pb-phase-caption .tag { display:inline-block; padding:3px 10px; border-radius:20px;
      font:700 11px/1 monospace; color:#fff; margin-bottom:8px; }
    .pb-phase-caption .desc { color:#c8c8d8; font-size:15px; line-height:1.5; }
    .pb-phase-caption .fields { margin-top:6px; color:#7c6fff; font:600 12px/1 monospace; }
  </style>
  <script>
    (function (root) {
      root.innerHTML = `
        <span class="tag" style="background:${root.dataset.color}">${root.dataset.phase}</span>
        <div class="desc">${root.dataset.description}</div>
        <div class="fields">${root.dataset.fields || ""}</div>`;
    })(document.currentScript.closest(".pb-phase-caption"));
  </script>
</div>
```
"""


def _build_html_user_prompt(topic: str, script_json: str) -> str:
    return f"""Topic: {topic}

Step-by-step script (JSON):
{script_json}

Build the animated explainer page for this topic following the system
instructions exactly. Decide what the PARTICIPANTS and MOVING ELEMENT
represent for this specific topic before writing any code."""


def _build_user_prompt(topic: str, target_steps: int = TARGET_STEPS, research_context: str = "") -> str:
    grounding_block = ""
    if research_context:
        grounding_block = f"""
## Retrieved research context (prefer these specific facts over general knowledge)

{research_context}

Use the above retrieved facts where relevant, but write the steps in your
own words — do not reproduce source text verbatim.

"""
    return f"""
{grounding_block}Generate a structured visual explainer for the following academic/technical topic:

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
        # Part 2 — research provenance
        "research_provider": "tavily" if payload.get("research_sources") else None,
        "research_sources": payload.get("research_sources", []),
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


# ── Research grounding (Part 2) ───────────────────────────────────────────────
def _research_topic(topic: str) -> tuple[str, list[str]]:
    """
    Search the web for the topic via Tavily → (context_block, source_urls).
    Returns ("", []) if TAVILY_API_KEY is unset or the search fails — the
    pipeline keeps working without grounding, just less accurately.
    """
    if not TAVILY_API_KEY:
        logger.info("TAVILY_API_KEY not set — skipping research step.")
        return "", []

    try:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=f"{topic} explained step by step technical",
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )
        results = response.get("results", [])
        if not results:
            return "", []

        context_lines = []
        sources = []
        if response.get("answer"):
            context_lines.append(f"Summary: {response['answer']}")
        for r in results:
            title = r.get("title", "")
            content = (r.get("content", "") or "")[:600]
            url = r.get("url", "")
            context_lines.append(f"- {title}: {content}")
            if url:
                sources.append(url)

        context_block = "\n".join(context_lines)
        logger.info("Research found %d sources for topic=%r", len(sources), topic)
        return context_block, sources

    except Exception as exc:  # noqa: BLE001
        logger.warning("Tavily research failed, continuing without grounding: %s", exc)
        return "", []


# ── HTML generation ────────────────────────────────────────────────────────────
def _generate_html(client, topic: str, script: LLMExplainerOutput) -> str | None:
    """
    Second LLM pass: given the validated script, ask the HTML model to produce
    a single self-contained animated HTML explainer page.
    Returns the raw HTML string, or None on failure (after one retry).
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
    user_prompt = _build_html_user_prompt(topic, script_json)

    def _strip_fences(raw: str) -> str:
        raw = raw.strip()
        for fence in ("```html", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):].lstrip()
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()
        return raw

    try:
        logger.info("Generating animated HTML for topic=%r using model=%s", topic, HTML_MODEL)
        response = client.chat.completions.create(
            model=HTML_MODEL,
            messages=[
                {"role": "system", "content": HTML_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=8192,
        )
        raw_output = response.choices[0].message.content or ""
        html = _strip_fences(raw_output)

        if html.lstrip().startswith("<!DOCTYPE"):
            logger.info("Animated HTML generated successfully (%d bytes)", len(html))
            return html

        # First attempt failed validation — retry with a stricter follow-up
        logger.warning("HTML model returned non-DOCTYPE output: %s", raw_output[:300])
        retry_response = client.chat.completions.create(
            model=HTML_MODEL,
            messages=[
                {"role": "system", "content": HTML_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": raw_output},
                {"role": "user", "content": (
                    "Your last response did not start with <!DOCTYPE html>. "
                    "Respond with ONLY the raw HTML document this time — "
                    "no commentary, no markdown fences."
                )},
            ],
            temperature=0.3,
            max_tokens=8192,
        )
        retry_raw = retry_response.choices[0].message.content or ""
        retry_html = _strip_fences(retry_raw)

        if retry_html.lstrip().startswith("<!DOCTYPE"):
            logger.info("Animated HTML generated on retry (%d bytes)", len(retry_html))
            return retry_html

        logger.warning("HTML model still did not return a DOCTYPE document after retry; discarding.")
        return None

    except Exception as exc:  # noqa: BLE001
        logger.error("HTML generation failed: %s", exc)
        return None


# ── HyperFrames composition HTML generation (Part 3, behind flag) ─────────────
def _generate_composition_html(client, topic: str, script: LLMExplainerOutput) -> str | None:
    """
    HyperFrames-aware HTML generation path — only called when
    USE_COMPOSITION_ENGINE=True. Uses an extended system prompt that
    instructs the model to structure output as a HyperFrames composition
    with clip divs, data-start/data-duration, and the three PulseBoard blocks.
    Falls back to returning None on any failure (caller will use existing path).
    """
    composition_system_prompt = HTML_SYSTEM_PROMPT + HYPERFRAMES_SYSTEM_PROMPT_ADDENDUM
    script_json = json.dumps(
        {
            "title": script.title,
            "category": script.category,
            "summary": script.summary,
            "steps": [s.model_dump() for s in script.steps],
        },
        indent=2,
    )
    user_prompt = _build_html_user_prompt(topic, script_json)

    def _strip_fences(raw: str) -> str:
        raw = raw.strip()
        for fence in ("```html", "```"):
            if raw.startswith(fence):
                raw = raw[len(fence):].lstrip()
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()
        return raw

    try:
        logger.info(
            "Generating HyperFrames composition HTML for topic=%r using model=%s",
            topic, HTML_MODEL,
        )
        response = client.chat.completions.create(
            model=HTML_MODEL,
            messages=[
                {"role": "system", "content": composition_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=8192,
        )
        raw_output = response.choices[0].message.content or ""
        html = _strip_fences(raw_output)

        if html.lstrip().startswith("<!DOCTYPE"):
            logger.info("HyperFrames composition HTML generated (%d bytes)", len(html))
            return html

        # Retry once
        logger.warning("Composition HTML model returned non-DOCTYPE output: %s", raw_output[:300])
        retry_response = client.chat.completions.create(
            model=HTML_MODEL,
            messages=[
                {"role": "system", "content": composition_system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": raw_output},
                {"role": "user", "content": (
                    "Your last response did not start with <!DOCTYPE html>. "
                    "Respond with ONLY the raw HTML document this time — "
                    "no commentary, no markdown fences."
                )},
            ],
            temperature=0.3,
            max_tokens=8192,
        )
        retry_html = _strip_fences(retry_response.choices[0].message.content or "")
        if retry_html.lstrip().startswith("<!DOCTYPE"):
            logger.info("HyperFrames composition HTML generated on retry (%d bytes)", len(retry_html))
            return retry_html

        logger.warning("Composition HTML still did not return DOCTYPE after retry; falling back.")
        return None

    except Exception as exc:  # noqa: BLE001
        logger.error("HyperFrames composition HTML generation failed: %s", exc)
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
    storage_configured = storage is not None
    run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc)
    last_error: Exception | None = None

    # Part 2 — Research grounding: call once before the retry loop
    research_context, research_sources = _research_topic(topic)
    research_used = bool(research_context)

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
                    {"role": "user", "content": _build_user_prompt(topic, research_context=research_context)},
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
                # Part 2 — research provenance
                "research_used": research_used,
                "research_sources": research_sources,
            }

            # Generate animated HTML (second LLM pass)
            # Part 0 — gate on ENABLE_COMPOSITION_ENGINE flag
            if USE_COMPOSITION_ENGINE:
                html_content = _generate_composition_html(client, topic, llm_output)
                if html_content is None:
                    # Composition path failed — fall back to standard path
                    logger.warning(
                        "Composition HTML generation failed; falling back to standard _generate_html."
                    )
                    html_content = _generate_html(client, topic, llm_output)
            else:
                html_content = _generate_html(client, topic, llm_output)

            html_generation_failed = html_content is None

            # Persist JSON + HTML + manifest to B2
            topic_slug = _slugify(topic)
            b2_url, manifest_url, html_url = _upload_to_b2(
                storage, run_id, topic_slug, payload, html_content
            )

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
                # Part 1 — diagnostic fields
                storage_configured=storage_configured,
                html_generation_failed=html_generation_failed,
                # Part 2 — research provenance
                research_used=research_used,
                research_sources=research_sources,
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
