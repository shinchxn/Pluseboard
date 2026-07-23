"""
pipeline.py — Core generation pipeline for PulseBoard.

Flow:
  1. Call NVIDIA NIM (script model) with a structured JSON prompt
  2. Parse + validate response against LLMExplainerOutput schema
  3. Accuracy-check the script with a second evaluator LLM call
  4. Retry up to MAX_RETRIES on JSON/schema failures OR accuracy failures
  5. Call NVIDIA NIM (html model) with the script to generate an animated HTML page
  6. Store the explainer JSON + animated HTML + provenance manifest to Backblaze B2 via genblaze-s3
  7. Return a populated Explainer dataclass to the API layer
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from openai import OpenAI
# pyrefly: ignore [missing-import]
from pydantic import ValidationError

from schemas import Explainer, LLMExplainerOutput, SlideStep

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Script model: reasoning-heavy, used for structured JSON script generation
SCRIPT_MODEL = os.getenv("PULSEBOARD_SCRIPT_MODEL", os.getenv("NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1"))
SCRIPT_MODEL_FALLBACKS = [
    SCRIPT_MODEL,
    "meta/llama-3.1-70b-instruct",
]
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
SYSTEM_PROMPT = """You are an expert academic curriculum designer and technical
illustrator producing step-by-step process explainers for computer science
classrooms. Precision matters more than polish — a professor will read this
aloud to students, so any factual error becomes a taught error.

BEFORE WRITING OUTPUT, silently work through these checks (do not include
this reasoning in your response, only the final JSON):
1. List the actual sequence of events for this topic, in strict causal order
   — each step must be a direct consequence of the step before it, not just
   a related fact placed nearby.
2. Identify anything you are not fully certain about. If uncertain, describe
   the mechanism at a level of detail you ARE certain of, rather than
   inventing specifics (exact byte offsets, exact constant names, exact
   version numbers) you are not sure of.
3. Check each step against this canonical accuracy pattern before finalizing:
   a correct process explainer describes STATE (what exists/is true at this
   point), ACTION (what specifically happens), and CONSEQUENCE (what state
   this produces for the next step) — not just a generic description of the
   topic area.

ACCURACY REQUIREMENTS:
- If retrieved research context is provided in the user message, treat it as
  higher-priority ground truth than your own training knowledge for specific
  facts (numbers, field names, protocol constants, algorithm names) — but
  still write in your own words, and still apply your own domain expertise
  to structure and explain it correctly.
- Do not pad steps with generic filler bullets ("this is important for
  performance", "this ensures reliability") that aren't specific to what's
  actually happening at that step — every bullet must state a concrete fact.
- Do not merge two distinct real-world steps into one, and do not split one
  real step into two just to hit a target step count — the step count exists
  to fit the topic, not the other way around.
- Prefer well-established terminology a professor would actually use in
  lecture over invented or overly casual phrasing.

All output must be valid, parseable JSON — nothing else, no explanation of
your reasoning process, only the final JSON object.
"""

EVALUATOR_SYSTEM_PROMPT = """You are a strict technical fact-checker reviewing
an educational script before it is shown to computer science students. You did
not write this script — review it skeptically, the way a professor reviewing
a teaching assistant's draft would.

Check specifically for:
- Factual errors (wrong terminology, wrong order of operations, wrong
  mechanism, incorrect numbers/constants/field names)
- Logical gaps — does each step actually follow causally from the one before it?
- Vague or generic bullets that don't state a real, specific fact
- Anything that contradicts the provided research context, if research context
  is present

Return ONLY a JSON object, nothing else:
{
  "accurate": <true if the script has no material errors, false otherwise>,
  "issues": ["<specific, actionable issue>", ...]
}

If accurate, return an empty issues list. Do not flag minor stylistic
preferences — only flag things that are actually wrong or actually missing,
not things you would have phrased differently.
"""

COMPOSITION_SYSTEM_PROMPT = """You generate ONE self-contained HTML document for an interactive
classroom visual explainer. Output ONLY the raw HTML — no markdown fences, no
commentary, no explanation. Your response must start with <!DOCTYPE html>.

STRUCTURE (required, exactly this shape):
- Root element: <div data-composition-id="pulseboard-explainer" data-width="1920" data-height="1080">
- One <div class="clip" data-start="{N}" data-duration="{D}"> per step, positioned
  sequentially on the timeline (step 0 starts at data-start="0"). D = 6 seconds
  per step unless the step has more than 3 bullets, then D = 8.
- Inside each step's clip: a phase label (e.g. "STEP 2 OF 4"), a title, a short
  description paragraph, and up to 3 bullet points — use the JSON step data
  provided in the user message for this content, do not invent facts.
- Exactly ONE moving/visual element that persists across all steps (e.g. a
  labeled packet, token, or cursor) that visually changes position, size, or
  attached labels as it moves through each step — this is the single most
  important visual, do not omit it.

VISUAL STYLE (required):
- Dark background (#0a0a0f or similar near-black navy), high contrast white/
  light-gray text, one accent color (#7c6fff purple) for active/highlighted
  elements — matches existing PulseBoard branding, do not invent a new palette.
- Large, legible type — this is viewed on a classroom smartboard from a
  distance, not a phone screen.

ANIMATION (required, do not use GSAP or any external library):
- Use native Web Animations API only: element.animate([...keyframes], {duration,
  fill: "forwards", easing: "ease-in-out"}). This keeps the file self-contained
  and lets it be seeked externally.
- Keep each individual animation under 1.5 seconds — the goal is a crisp state
  change per step, not a long cinematic transition.

INTERACTION (required — this must work standalone, without any external player):
- One JS function `renderStep(index)` that sets the visual state for a given
  step index directly (not via animation replay) — this is the single source
  of truth both for click navigation and for external seeking.
- Next/Previous buttons that call renderStep(currentStep ± 1).
- Arrow-key (ArrowRight/ArrowLeft) keyboard support calling the same function.
- A progress-dots row showing current step, updated by the same function.

SCOPE — keep this achievable in one response:
- Do not add sound, video, external fonts, external images, or CDN scripts of
  any kind. Everything must be inline <style> and inline <script> in one file.
- Prioritize correctness and a working moving element over visual complexity.
  A simple, reliable animation beats an elaborate one that risks malformed output.
"""

HTML_SYSTEM_PROMPT = """You are a front-end engineer building a single, self-contained animated
HTML page for PulseBoard — a classroom smartboard tool that replaces static
PPT/PDF slides with a live, tap-to-advance visual simulation of a process.

You will be given a topic and a step-by-step script. Your job is NOT to lay
that content out as text cards or bullet slides. Build a real simulation
using a proper visualization library suited to the topic — do not hand-roll
animation logic from scratch when a library does it more reliably.

## Step 1 — choose your library based on the topic's shape

- Two-party exchange, protocol, pipeline, queue, state machine (handshakes,
  request/response, message passing) → GSAP, via
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
  Use gsap.to(element, {left: '92%', duration: 1.1, ease: 'power2.inOut'})
  to move elements — never hand-write transition/requestAnimationFrame code
  when GSAP is loaded.

- Algorithm on an array/tree/graph (sorting, search, traversal, trie/automaton
  construction, pathfinding) → D3.js v7, via
  <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
  Bind your step data to DOM elements with d3.select/.data()/.join(), and
  animate attribute changes with .transition().duration(800).attr(...).

- Physics, molecules, 3D/spatial structures → Three.js r128, via
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  Set up a scene/camera/renderer, animate via requestAnimationFrame updating
  object.position or object.rotation each frame. r128 has no OrbitControls
  — implement basic drag-to-rotate manually if needed. Do not use
  CapsuleGeometry (needs r142+); use Sphere/Cylinder/custom geometry instead.

- Statistics, distributions, trends → Chart.js, via
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
  Use chart.update() with new data per step so the chart animates its own
  transition — do not hand-animate bars/points yourself.

- Geography/maps → Leaflet, via the CSS + JS tags for leaflet 1.9.4.

- If the topic is a single, simple diagram that doesn't fit any of the
  above (e.g. a static system diagram with one highlighted element per
  step) → raw inline SVG with CSS `transition`/`@keyframes`, no library
  needed. Don't force a library where plain SVG is clearly simpler.

## Step 2 — the architecture, regardless of library

- ONE JavaScript array `steps[]` holding each step's data (participant
  states, what moves/changes, caption text, live field values).
- ONE `render(stepIndex)` function that reads from steps[] and drives the
  chosen library's update/transition/animate call for that step.
- Never hand-write repeated per-step markup — the library call inside
  render() should be the only place visuals change.
- Every step must be replayable: calling render(i) again re-triggers that
  step's motion, not just its end state.

## Step 3 — the five required pieces (unchanged regardless of library)

1. PARTICIPANTS — labeled boxes/nodes/elements with a live, JS-updated
   state label.
2. THE MOVING ELEMENT — whatever the chosen library animates between
   states each step, carrying the concrete value involved (not just a
   flag name).
3. LIVE FIELDS — any changing number/value rendered via the library's own
   update mechanism or `.textContent =`, never static HTML text.
4. STEP CAPTION — one color-coded tag + 1-2 sentences of prose, the ONLY
   place prose appears.
5. CONTROLS — Play/Pause, Next, Reset, progress dots, spacebar shortcut,
   44px+ touch targets.

## Non-negotiable check before you respond

1. Confirm you actually called the chosen library's animation/update API
   (gsap.to / d3 .transition() / chart.update() / Three.js position update
   in a render loop) — not a manual CSS transition you wrote by hand
   instead of using the library you loaded.
2. Confirm at least one value updates via JS on every step, not static HTML.
3. If your render() only swaps caption text with no library call changing
   any visual element's position/data/state — you've built a slideshow.
   Fix it before responding.

Output ONE complete HTML document. External <script>/<link> tags are
allowed ONLY for the specific library CDN URLs listed above — no other
external network calls. Respond with ONLY the raw HTML document, starting
with <!DOCTYPE html>. No markdown fences, no commentary.
"""


def _build_html_user_prompt(topic: str, script_json: str) -> str:
    return f"""Topic: {topic}

Step-by-step script (JSON):
{script_json}

Build the animated explainer page for this topic following the system
instructions exactly. Decide what the PARTICIPANTS and MOVING ELEMENT
represent for this specific topic before writing any code."""


def _build_user_prompt(
    topic: str,
    target_steps: int = TARGET_STEPS,
    research_context: str = "",
    accuracy_issues: list[str] | None = None,
) -> str:
    grounding_block = ""
    if research_context:
        grounding_block = f"""
## Retrieved research context (prefer these specific facts over general knowledge)

{research_context}

Use the above retrieved facts where relevant, but write the steps in your
own words — do not reproduce source text verbatim.

"""
    correction_block = ""
    if accuracy_issues:
        issues_text = "\n".join(f"- {issue}" for issue in accuracy_issues)
        correction_block = f"""
## Fix these specific accuracy issues from the previous attempt

{issues_text}

"""
    return f"""
{grounding_block}{correction_block}Generate a structured visual explainer for the following academic/technical topic:

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
        timeout=90.0,
        max_retries=0,  # We handle retries manually; don't stack internal retries
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
        # Accuracy verification result
        "accuracy_verified": payload.get("accuracy_verified", False),
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

        page = storage.list(prefix="explainers/")
        for entry in page.entries:
            key = entry.key
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
                    "html_url": data.get("html_url"),
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
        page = storage.list(prefix="explainers/")
        for entry in page.entries:
            if explainer_id in entry.key:
                target_key = entry.key
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


# ── Accuracy evaluator (generate → evaluate → retry) ─────────────────────────
def _evaluate_script_accuracy(
    client, topic: str, llm_output: LLMExplainerOutput, research_context: str
) -> tuple[bool, list[str]]:
    """
    Fact-check the generated script before proceeding to HTML generation.
    Returns (accurate, issues). On any failure, returns (True, []) so a broken
    evaluator never blocks an otherwise-fine topic.
    """
    script_json = json.dumps(
        {
            "title": llm_output.title,
            "steps": [s.model_dump() for s in llm_output.steps],
        },
        indent=2,
    )
    research_block = (
        f"\n\nResearch context used:\n{research_context}" if research_context else ""
    )

    try:
        response = client.chat.completions.create(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Topic: {topic}\n\nScript to review:\n{script_json}{research_block}",
                },
            ],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = json.loads(raw)
        accurate = bool(result.get("accurate", True))
        issues = result.get("issues", []) or []
        if not accurate:
            logger.warning(
                "Accuracy check flagged issues for topic=%r: %s", topic, issues
            )
        return accurate, issues

    except Exception as exc:  # noqa: BLE001
        logger.warning("Accuracy evaluator failed, treating as accurate: %s", exc)
        return True, []


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

        if html.lstrip().lower().startswith("<!doctype"):
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

        if retry_html.lstrip().lower().startswith("<!doctype"):
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
    composition_system_prompt = COMPOSITION_SYSTEM_PROMPT
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

        if html.lstrip().lower().startswith("<!doctype"):
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
        if retry_html.lstrip().lower().startswith("<!doctype"):
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
    Full generate → evaluate → retry → B2-store pipeline.
    Returns a populated Explainer on success, raises GenerationError on failure.
    """
    client = _get_nim_client()
    storage = _get_storage_backend()
    storage_configured = storage is not None
    run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc)
    last_error: Exception | None = None
    accuracy_issues: list[str] = []   # accumulated from the evaluator across attempts
    accuracy_verified = False

    # Part 2 — Research grounding: call once before the retry loop
    research_context, research_sources = _research_topic(topic)
    research_used = bool(research_context)

    for attempt in range(1, MAX_RETRIES + 1):
        # Slightly raise temperature on retries to escape stuck outputs
        temperature = 0.2 + (attempt - 1) * 0.15
        model_for_attempt = SCRIPT_MODEL_FALLBACKS[
            min(attempt - 1, len(SCRIPT_MODEL_FALLBACKS) - 1)
        ]
        logger.info(
            "Generation attempt %d/%d for topic=%r using model=%s (temp=%.2f)",
            attempt, MAX_RETRIES, topic, model_for_attempt, temperature,
        )
        script_max_tokens = 2048 if attempt == 1 else 1024

        try:
            response = client.chat.completions.create(
                model=model_for_attempt,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(
                        topic,
                        research_context=research_context,
                        accuracy_issues=accuracy_issues,
                    )},
                ],
                temperature=temperature,
                max_tokens=script_max_tokens,
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

            # Accuracy check — fact-check the script before moving to HTML
            is_accurate, issues = _evaluate_script_accuracy(
                client, topic, llm_output, research_context
            )
            if not is_accurate and attempt < MAX_RETRIES:
                accuracy_issues = issues
                logger.warning(
                    "Attempt %d failed accuracy check, retrying with corrections: %s",
                    attempt, issues,
                )
                continue  # skip to next attempt with issues injected into prompt
            accuracy_verified = is_accurate

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
                # Accuracy verification result
                "accuracy_verified": accuracy_verified,
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
                # Accuracy verification
                accuracy_verified=accuracy_verified,
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
