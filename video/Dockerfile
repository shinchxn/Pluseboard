# PulseBoard HyperFrames vendor — Dockerfile
#
# Based on the official Puppeteer image (Chromium + Node bundled).
# Pin a specific digest/tag once you've verified which version works with your
# HyperFrames fork — never use :latest in production.
#
# To build: docker build -t pulseboard-hyperframes .
# To run rendering (example): docker run --rm -v $(pwd)/out:/app/out pulseboard-hyperframes

FROM ghcr.io/puppeteer/puppeteer:22.15.0

# Switch to root to install system deps
USER root

# Install FFmpeg (needed for any future MP4 render — not required for live playback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Drop back to the puppeteer user for security
USER pptruser

WORKDIR /app

# Copy manifests first so layer is cached on dep changes
COPY --chown=pptruser:pptruser package.json bun.lockb* package-lock.json* ./

# Install deps. Prefer bun if lockfile is present; fall back to npm.
RUN if [ -f bun.lockb ]; then \
      npm install -g bun && bun install --frozen-lockfile; \
    else \
      npm ci --omit=dev; \
    fi

# Copy the rest of the vendor source
COPY --chown=pptruser:pptruser . .

# Default command — can be overridden for one-shot renders
CMD ["node", "--version"]
