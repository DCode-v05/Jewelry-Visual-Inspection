# Jewelry Visual Inspection

**Reference-based automated QC for jewelry: compare a master image to a freshly made piece and get an ACCEPT / REVIEW / REJECT verdict in under a second, fully offline.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/DINOv2-FFD21E?style=flat&logo=huggingface&logoColor=black) ![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black) ![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white) ![Vite](https://img.shields.io/badge/Vite-5-646CFF?style=flat&logo=vite&logoColor=white)

## Overview

On a jewelry production line, every piece is supposed to match an approved reference: the right shape, every stone present and the correct color, and a clean surface with no scratches, pits, or tarnish. Doing that check by eye is slow and inconsistent — two operators will disagree, and a tired one at hour six disagrees with themselves.

This project automates that comparison. You give it a *master* image (the golden reference for a SKU) and a *live* image of a newly produced piece, and it returns one of three verdicts — **ACCEPT**, **REVIEW**, or **REJECT** — in well under a second, entirely on-device with no cloud calls. Under the hood it runs an 8-stage computer-vision pipeline that first aligns the live piece to the master, then judges it across three independent checks (profile, decoration, surface) and draws a labeled box around every defect it finds.

I built this during my time at iQube as a reference-quality QC prototype. The hard part isn't spotting a scratch in isolation — it's that the live piece is photographed at an arbitrary rotation and a slightly different scale, so before any pixel-level comparison means anything, the specimen has to be segmented, rotated into the master's frame, and finely registered. Only then can a per-region difference be blamed on a real defect instead of on misalignment. Most of the engineering here is in getting that alignment trustworthy enough that the judging stages can be strict.

## Key Features

- **Three-way verdict, not a binary** — ACCEPT / REVIEW / REJECT. Borderline pieces route to REVIEW (a human) instead of being auto-passed, which is what keeps false accepts near zero.
- **Three independent checks**, each answering a different question:
  - *Profile* — is the shape right? (outline, prongs, cutouts, engraving structure)
  - *Decoration* — is everything present and the right color? (stones, pavé, engraving)
  - *Surface* — is the finish clean? (scratches, pits, tarnish, polishing marks)
- **Defect localization** — every flagged issue gets a color-coded bounding box (red = profile, orange = decoration, magenta = surface); scratches are drawn as rotated polygons, not just axis-aligned rectangles.
- **Rotation-tolerant alignment** — handles arbitrary in-plane rotation and scale drift via moment-based estimation plus SIFT/RANSAC fine registration, and resolves the 180-degree moment ambiguity with masked NCC.
- **Foundation model + classical CV mix** — DINOv2 patch features for "is this stone present / correct" combined with CIELAB delta-E for color, plus classical Otsu/Canny/Hu-moments/SSIM for everything else.
- **Rich visual report** — master with drawn edge contours, a left-to-right strip of the live image through each alignment stage, per-check heatmaps, and a per-stage timing row so latency budgets are visible.
- **Offline / on-device** — no external services. DINOv2 runs on GPU automatically when CUDA is available, CPU otherwise.
- **FastAPI service** — three endpoints (`/api/healthz`, `/api/info`, `/api/inspect`) with model warmup on startup.
- **Editorial React UI** — drag-and-drop upload, a pre-rotation control, the verdict banner, per-check result cards, and an instrument panel fed live from the backend's `/api/info`.
- **No training step** — the system is reference-based and deterministic. Behavior is governed by documented per-check thresholds meant to be calibrated per SKU, not learned weights you have to retrain.

## How It Works

Each inspection runs the live image through eight stages in order, with per-stage timing captured throughout. The first four **align** the piece; the last four **judge** it. The orchestrator (`backend/pipeline/orchestrator.py`) chains the stages and the API serializes the result, every intermediate image, and the timings back to the frontend as base64 PNGs.

### Stage 1-4: Align the piece

1. **Preprocess** — decode the image and resize its long side to a 1024 px working resolution, keeping a full-resolution copy around for later.
2. **Segmentation** — Otsu threshold with automatic background-polarity detection, morphological cleanup, and a largest-connected-component step to produce a binary mask, bounding box, and centroid. The piece is tight-cropped and the live crop is resized to the master's frame.
3. **Rotation estimation** — a single similarity transform aligns the live piece to the master: rotation from image moments, translation centroid-to-centroid, isotropic scale from the square root of area ratio. The 180-degree moment ambiguity is broken by keeping whichever of the two candidate orientations scores higher masked NCC.
4. **Fine registration** — SIFT keypoints with Lowe's ratio test and a FLANN matcher, then RANSAC homography to warp the live image into the master frame. It reports NCC, inlier count, and a reliability flag. If rotation alone already reaches NCC >= 0.92, SIFT is skipped to save time.

### Stage 5-7: Judge it

5. **Profile check** — compares Canny edge structure via Hu-moment shape distance, edge-area deviation, dilated edge IoU, and a missing-edge ratio. Flags MISSING (red) and EXCESS (orange) regions.
6. **Decoration check** — DINOv2 (ViT-S/14, `facebook/dinov2-small`) per-patch cosine similarity on a 16x16 grid, combined with per-patch CIELAB delta-E color comparison. The patch features catch a missing or chipped stone; the color term catches a *swapped* stone color (a ruby where a diamond belongs) that semantic features alone tend to underweight.
7. **Surface check** — fuses SSIM, a Laplacian high-frequency difference, and a per-pixel CIELAB color anomaly into one defect map. It separates smooth zones (adaptive median + MAD threshold) from textured stone zones (color-only, so specular highlights inside a gem don't false-trigger), then classifies scratches (rotated boxes), pits, and stone defects.

### Stage 8: Decide

The pipeline uses two label vocabularies. Each of the three checks returns **PASS / BORDERLINE / FAIL** on its own metrics; the decision engine (`decision.py`) fuses those plus the registration reliability into the final verdict:

- **REJECT** — any check is FAIL.
- **REVIEW** — registration is unreliable (`inliers < 15` or `inlier_ratio < 0.3`), *or* any check is BORDERLINE.
- **ACCEPT** — all three checks PASS and registration is reliable.

Each verdict comes with human-readable reasons (e.g. "surface: 2 scratches detected", "registration unreliable — only 9 inliers").

### Thresholds and calibration

There's no model training. Behavior is set by per-check thresholds that are meant to be calibrated on a labeled validation set per SKU rather than tuned at runtime. The defaults shipped in the pipeline:

```
Profile      shape_dist    FAIL >0.30  / BORDER >0.15
             area_dev      FAIL >0.40  / BORDER >0.20
             edge_IoU      FAIL <0.30  / BORDER <0.50
             missing_edge  FAIL >0.40  / BORDER >0.20

Decoration   global_sim    FAIL <0.85  / BORDER <0.90
             problem_ratio FAIL >0.10  / BORDER >0.04
             color dE      FAIL >25    / BORDER >12   (patch sim threshold 0.75)

Surface      defect_ratio  FAIL >0.012 / BORDER >0.004
             color dE      FAIL >30    / BORDER >15
             any MAJOR / SCRATCH / STONE defect → FAIL

Registration reliable when inliers >= 15 AND inlier_ratio >= 0.3
Rotation     skip SIFT when masked NCC >= 0.92
```

### The web app

The Vite + React + TypeScript frontend is built around a `/api/inspect` round trip. It has drag-and-drop upload zones for the master and live images, a rotation control to roughly pre-orient the live piece before inspection, and a **Run Verification** button that streams both images as multipart to the backend. The response drives a verdict banner with reasons, the full visual display (master with contours, the live stage-progression strip, per-check heatmaps, color-coded boxes), per-check result cards with the underlying metrics, and a per-stage timing row. An Instrument Panel reads `/api/info` to show the active model name, device (CUDA/CPU), working resolution, and the list of pipeline stages. In development the Vite server proxies `/api` to the backend on port 8000.

## Results / Highlights

These are the documented acceptance targets from the validation plan. They are design targets, not benchmarked numbers — the repo ships the pipeline and sample images but no measured benchmark artifact:

- **Speed:** total pipeline **P95 < 1000 ms**.
- **Accuracy:** **0% false accept** on the test set — a defective piece must never be auto-accepted.
- **Localization:** **defect-localization recall > 90%**.

The design choice that backs the zero-false-accept target is that anything ambiguous (unreliable registration, or any single BORDERLINE check) is forced to REVIEW rather than ACCEPT, trading some throughput for the guarantee that questionable pieces always reach a human.

## Tech Stack

- **Languages:** Python (>= 3.10), TypeScript, CSS
- **Backend / API:** FastAPI, Uvicorn, Pydantic, python-multipart
- **Computer vision / ML:** OpenCV (`opencv-contrib-python`, for SIFT), PyTorch + Torchvision, Hugging Face Transformers (DINOv2 `facebook/dinov2-small`), scikit-image (SSIM, histogram matching), SciPy, NumPy, Pillow
- **Frontend:** React 18, TypeScript 5.5, Vite 5, `@vitejs/plugin-react`
- **Compute:** runs on CPU; uses CUDA automatically for DINOv2 when a GPU is available

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Node.js 18+ and npm (for the frontend)
- An optional CUDA-capable GPU for faster DINOv2 inference

### Installation

```bash
git clone https://github.com/DCode-v05/Jewelry-Visual-Inspection.git
cd Jewelry-Visual-Inspection

# Backend
python -m venv env
# Windows:
env\Scripts\activate
# macOS / Linux:
source env/bin/activate

pip install -r requirements.txt
# torch is installed separately (CPU shown; use the CUDA index URL for GPU):
pip install torch torchvision
# CUDA example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Running

```bash
# Backend API (from the repo root)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (in a second terminal)
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api to :8000
```

The first request after startup pays the DINOv2 load cost; startup runs a best-effort warmup to absorb most of it.

### API endpoints

- `GET  /api/healthz` — liveness probe.
- `GET  /api/info` — model name, device, working resolution, pipeline stages, and the bounding-box color legend (used by the Instrument Panel).
- `POST /api/inspect` — multipart form with `master` and `live` images; returns the verdict, per-check metrics and boxes, every stage image as a base64 PNG, and the timings.

## Usage

1. Start the backend (`uvicorn`) and the frontend (`npm run dev`), then open **http://localhost:5173**.
2. Upload a **master** (golden reference) image and a **live** specimen image.
3. Optionally use the rotation control to roughly orient the live piece — the pipeline handles fine alignment, this just gives it a good starting point.
4. Click **Run Verification**. The live image goes through all eight stages.
5. Read the verdict (ACCEPT / REVIEW / REJECT), inspect the per-check heatmaps and bounding boxes, and check the per-stage timing.

For consistent results, capture master and live images under the same lighting and camera setup against a matte background, and save as lossless PNG — JPEG compression artifacts can either mask real defects or fake them.

## Project Structure

```
Jewelry-Visual-Inspection/
├── backend/                         # FastAPI service + 8-stage pipeline
│   ├── main.py                      # API: /api/healthz, /api/info, /api/inspect; CORS; warmup
│   └── pipeline/
│       ├── preprocess.py            # Stage 1 — load & resize to working resolution
│       ├── segmentation.py          # Stage 2 — Otsu mask, bbox, centroid, crop
│       ├── rotation.py              # Stage 3 — moment-based similarity alignment
│       ├── registration.py          # Stage 4 — SIFT + FLANN + RANSAC homography
│       ├── alignment.py             # shared alignment / warp helpers
│       ├── profile_check.py         # Stage 5 — Hu moments, edge IoU, area deviation
│       ├── decoration_check.py      # Stage 6 — DINOv2 patch similarity + LAB deltaE
│       ├── surface_check.py         # Stage 7 — SSIM + Laplacian + color anomaly
│       ├── decision.py              # Stage 8 — ACCEPT / REVIEW / REJECT fusion
│       ├── orchestrator.py          # runs all stages, captures per-stage timing
│       ├── visualizer.py            # edge overlays, heatmaps, bounding boxes
│       └── types.py                 # shared dataclasses for results
│
├── frontend/                        # Vite + React + TypeScript UI
│   ├── index.html
│   ├── vite.config.ts               # dev server + /api proxy
│   └── src/
│       ├── App.tsx, main.tsx        # app shell + entry
│       ├── api.ts, types.ts         # backend client + shared types
│       ├── components/              # Masthead, Hero, UploadRow, Dropzone, RotationControl,
│       │                            # Verdict, VisualDisplay, CheckResults, TimingRow,
│       │                            # InstrumentPanel, Divider, Footer
│       ├── utils/rotate.ts          # client-side image pre-rotation
│       └── styles/app.css           # editorial "onyx & gold" theme
│
├── Images/                          # sample master / defect test images
├── requirements.txt                 # backend Python dependencies
└── README.md
```

---

## Contact

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
