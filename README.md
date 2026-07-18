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
