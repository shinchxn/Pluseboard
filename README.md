# PulseBoard

PulseBoard is an interactive visual explainer tool designed for higher-education classroom smartboards. It empowers professors to turn any complex process-flow topic into custom, high-contrast, touch-optimized visual presentations on the fly.

## Project Scope

This codebase contains the complete **Landing & Input Screen** of PulseBoard. Designed to be the very first view a professor sees before a lecture begins, this interface serves as the portal for custom topic generation and presentation management.

### Key Visual and Interaction Features

1. **Header & Tagline**: High-contrast, elegant branding featuring custom typographic layout (Space Grotesk + Inter) and an academic aesthetic optimized for large projectors and smartboards.
2. **Dynamic Natural Language Input**: A prominent search/topic input field where professors can specify any technical sequence or flowchart topic (e.g., "OSI Model Packet Flow", "TCP Handshake", "Dijkstra's Shortest Path").
3. **Suggested Topic Chips**: A collection of cross-discipline quick-select suggestion pills representing real computer science curriculum (networking, operating systems, databases, and algorithms). Clicking any suggestion pre-fills the input field instantly.
4. **Active Primary Call to Action**: An extra-large, finger-friendly primary "Generate Explainer" button (56px minimum height) that unlocks dynamic micro-animations once text is entered.
5. **Interactive Saved Library**:
   - Includes realistic academic topic placeholders representing a professor's accumulated course materials.
   - Categorized badges for visual rhythm.
   - An interactive toggle to simulate a **First-Time Empty Library State** with guidance illustrations, allowing testing of both user conditions.
6. **Live Classroom Simulation Mode**:
   - **Generation Loader**: Interactive multi-stage progression representing real logical compilation.
   - **Interactive Presentation Preview**: Tapping "View Explainer" launches a full-screen smartboard visual mockup. It features responsive side-stepping controls (large touch targets), step progress indicators, and custom process descriptions for the active topic.

---

## Technical Stack

- **Framework**: React 19 (TypeScript)
- **Build System**: Vite
- **Styling**: Tailwind CSS v4.0
- **Animations**: Motion (formerly Framer Motion)
- **Icons**: Lucide React

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) (v18+)
- npm or yarn

### Installation

1. Clone or extract the project directory.
2. Install the necessary packages:
   ```bash
   npm install
   ```

### Running Development Server

To boot the dev server on port `3000` (optimized for AI Studio container configuration):
```bash
npm run dev
```

### Production Build

To compile and optimize the client-side single-page application into the `dist/` directory:
```bash
npm run build
```

---

## Backend — FastAPI Pipeline (New)

The real backend replaces the React prototype's simulation with live NVIDIA NIM generation and Backblaze B2 storage. It lives in `backend/`.

### Stack

| Layer | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| LLM inference | NVIDIA NIM via `genblaze-nvidia` |
| Storage / provenance | Backblaze B2 via `genblaze-s3` |
| Schema validation | Pydantic v2 |

### Pipeline

```
POST /api/generate  { topic }
  → NVIDIA NIM (llama-3.3-70b-instruct)
  → JSON schema validation (Pydantic) + retry loop (up to 3×)
  → Store explainer JSON + provenance manifest to B2
  → Return Explainer with b2_url + manifest_url

GET  /api/library   → List all explainers from B2
GET  /api/explainer/{id}  → Fetch one by run_id
```

### Backend Setup

**Prerequisites**: Python 3.11+

```bash
cd backend

# 1. Copy and fill in your credentials
cp .env.example .env
#    Edit .env — add NVIDIA_API_KEY, B2_KEY_ID, B2_APP_KEY, B2_BUCKET_NAME, B2_ENDPOINT_URL

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs** (Swagger UI).

### Frontend Setup

The frontend is a React application built with Vite.

```bash
# 1. Install dependencies
npm install

# 2. Start the dev server
npm run dev
# Open http://localhost:3000
```

> **Note**: The FastAPI server must be running on `http://localhost:8000` for the frontend to work. Set `VITE_API_URL` in root `.env` or `.env.local` if your backend is hosted elsewhere.

### Environment Variables (backend/.env)

| Variable | Description |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM API key from [build.nvidia.com](https://build.nvidia.com/) |
| `NVIDIA_BASE_URL` | NIM endpoint (default: `https://integrate.api.nvidia.com/v1`) |
| `NVIDIA_MODEL` | Model to use (default: `meta/llama-3.3-70b-instruct`) |
| `B2_KEY_ID` | Backblaze B2 application key ID |
| `B2_APP_KEY` | Backblaze B2 application key secret |
| `B2_BUCKET_NAME` | B2 bucket name (must exist) |
| `B2_ENDPOINT_URL` | B2 S3-compatible endpoint URL |
| `MAX_RETRIES` | LLM retry attempts on schema failure (default: `3`) |
| `TARGET_STEPS` | Number of slides to generate (default: `4`) |
| `ENVIRONMENT` | `development` enables CORS wildcard + verbose errors |

---

## Hackathon

Built for the **Backblaze Generative Media Hackathon** (Build with Genblaze on B2) — deadline August 4, 2026.

- **Genblaze tooling**: `genblaze-core`, `genblaze-s3` (for all B2 operations), `genblaze-nvidia` (for structured LLM generation)
- **Storage**: Backblaze B2 with provenance manifests per generation via `genblaze-s3`
- **Model**: NVIDIA NIM (`meta/llama-3.3-70b-instruct`) using `genblaze_nvidia`'s schema enforcement.
