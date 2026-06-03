# Jewelry Visual Inspection

## Project Description
This project is a **reference-based visual inspection system** for jewelry. It compares a *master* image (the golden reference for a SKU) against a *live* image of a newly produced piece and returns an **ACCEPT / REVIEW / REJECT** decision in well under a second — entirely offline and on-device. The system runs an 8-stage computer-vision pipeline that aligns the live piece to the master and judges it across three independent checks (profile, decoration, and surface), localizing every defect with a labeled bounding box. The solution ships as a FastAPI backend and an interactive React web application with an editorial "onyx & gold" interface.

---

## Project Details

### Problem Statement
On a jewelry production line, every piece must match an approved reference: correct shape, all stones present and the right color, and a clean surface free of scratches, pits, and tarnish. Manual visual QA is slow, subjective, and inconsistent between operators. This system automates that comparison. The hard part is that the live piece is photographed at an arbitrary rotation and slightly different scale, so before anything can be compared, the specimen must be segmented from the background, rotated into the master's frame, and finely registered — only then can per-region differences be attributed to real defects rather than misalignment.

### The Pipeline (8 Stages)
Each inspection runs the live image through eight stages in order, with per-stage timing captured throughout. The first four **align** the piece; the last four **judge** it.

| # | Stage | What it does |
|---|-------|--------------|
| 1 | **Preprocess** | Decodes the image and resizes its long side to a 1024 px working resolution, keeping a full-resolution copy. |
| 2 | **Segmentation** | Otsu threshold with auto background-polarity detection + morphology + largest connected component → binary mask, bounding box, centroid. The piece is tight-cropped, and the live crop is resized to the master's frame. |
| 3 | **Rotation estimation** | Aligns the live piece to the master with a single similarity transform (rotation from image moments, translation centroid→centroid, isotropic scale from √area). The 180° moment ambiguity is resolved by keeping whichever candidate scores higher masked NCC. |
| 4 | **Fine registration** | SIFT keypoints + Lowe's ratio test + RANSAC homography (FLANN matcher). Warps the live into the master frame and reports NCC, inlier count, and a reliability flag. If rotation alone already achieves NCC ≥ 0.92, SIFT is skipped to save time. |
| 5 | **Profile check** | Compares Canny edge structure (outline + pavé + cutouts + engravings) via Hu-moment shape distance, edge-area deviation, dilated edge IoU, and missing-edge ratio. Flags **MISSING** (red) and **EXCESS** (orange) regions. |
| 6 | **Decoration check** | DINOv2 (ViT-S/14) per-patch cosine similarity on a 16×16 grid, plus per-patch CIELAB ΔE color comparison. Catches missing/chipped/wrong-color stones and engraving mismatches. |
| 7 | **Surface check** | Fuses SSIM, Laplacian high-frequency diff, and per-pixel CIELAB color anomaly into a defect map. Separates smooth zones (adaptive threshold) from textured stone zones (color-only). Classifies scratches (rotated boxes), pits, and stone defects. |
| 8 | **Decision** | Fuses the three verdicts + registration reliability into one ACCEPT / REVIEW / REJECT outcome with human-readable reasons. |

### The Three Checks
- **Profile** — *Is the shape right?* Edge-based comparison catches wrong size, bent prongs, missing structural elements, and asymmetry.
- **Decoration** — *Is everything where it should be, and the right color?* DINOv2 patch features catch missing/chipped stones and misaligned engraving; LAB ΔE catches a swapped stone color (e.g., a ruby where a diamond belongs) that semantic features alone may underweight.
- **Surface** — *Is the finish clean?* Scratches, pits, tarnish spots, polishing marks, and surface contamination — including a separate, more tolerant color check inside textured stone regions so specular highlights don't false-trigger.

### Decision Engine
The pipeline emits two label vocabularies. Each check returns **PASS / BORDERLINE / FAIL**; the decision engine fuses them into:

- **REVIEW** — registration is unreliable (`inliers < 15` or `inlier_ratio < 0.3`), *or* any check is BORDERLINE.
- **REJECT** — any check is FAIL.
- **ACCEPT** — all three checks PASS.

### Thresholds & Calibration
There is no model training step — the pipeline is reference-based and deterministic. Behavior is governed by per-check thresholds that are meant to be calibrated on a labeled validation set per SKU (not tuned at runtime). Key defaults:

```
Profile      shape_dist   FAIL >0.30  / BORDER >0.15
             area_dev     FAIL >0.40  / BORDER >0.20
             edge_IoU     FAIL <0.30  / BORDER <0.50
             missing_edge FAIL >0.40  / BORDER >0.20

Decoration   global_sim   FAIL <0.85  / BORDER <0.90
             problem_ratio FAIL >0.10 / BORDER >0.04
             color ΔE     FAIL >25    / BORDER >12   (patch sim threshold 0.75)

Surface      defect_ratio FAIL >0.012 / BORDER >0.004
             color ΔE     FAIL >30    / BORDER >15
             any MAJOR / SCRATCH / STONE defect → FAIL

Registration reliable when inliers ≥ 15 AND inlier_ratio ≥ 0.3
Rotation     skip SIFT when masked NCC ≥ 0.92
```

### Visualizations
Every inspection produces a rich visual report:
- **Master panel** — the reference plus its drawn edge contours.
- **Live stage progression** — preprocessed → segmented (mask overlay) → rotated → SIFT-registered, shown left to right.
- **Per-check heatmaps** — profile deviation overlay, DINOv2 + color decoration heatmap, and surface JET defect map (rendered as "anomaly above the noise floor" so clean areas stay dark).
- **Color-coded bounding boxes** — red = profile, orange = decoration, magenta = surface, with green/yellow outlines marking the master/live silhouettes. Scratches are drawn as rotated polygons.
- **Per-stage timing row** — millisecond cost of every stage so latency budgets can be verified.

### Web Application
The Vite + React + TypeScript app provides:
- Drag-and-drop upload zones for the **master** and **live** images.
- A rotation control to pre-rotate the live image before inspection.
- A **Run Verification** button that streams both images to the backend.
- A verdict banner (ACCEPT / REVIEW / REJECT) with reasons.
- The full visual display, per-check result cards, and per-stage timing.
- An **Instrument Panel** populated from `/api/info` (model name, device, working resolution, pipeline stages).

---

## Tech Stack
- **Backend:** Python ≥ 3.10, FastAPI, Uvicorn, Pydantic, python-multipart
- **Computer vision / ML:** OpenCV (`opencv-contrib-python`, for SIFT), PyTorch + Torchvision, Hugging Face Transformers (DINOv2 `facebook/dinov2-small`), scikit-image (SSIM, histogram matching), SciPy, NumPy, Pillow
- **Frontend:** React 18, TypeScript, Vite, `@vitejs/plugin-react`
- **CUDA (optional):** the DINOv2 model runs on GPU automatically when available, CPU otherwise

---

## Getting Started

### 1. Clone the repository
```
git clone https://github.com/DCode-v05/Jewelry-Visual-Inspection.git
cd Jewelry-Visual-Inspection
```

### 2. Set up the backend
```
python -m venv env
# Windows:
env\Scripts\activate
# macOS / Linux:
source env/bin/activate

pip install -r requirements.txt
# Install torch separately (CPU shown; use the CUDA index URL for GPU):
pip install torch torchvision
# CUDA example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Run the API:
```
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET  /api/healthz` — liveness probe
- `GET  /api/info`    — model + pipeline info for the Instrument Panel
- `POST /api/inspect` — multipart `master`, `live` → full inspection result

### 3. Set up the frontend
```
cd frontend
npm install
npm run dev      # http://localhost:5173 (proxies /api to :8000)
```

The dev server proxies `/api` to `http://localhost:8000` (configured in `frontend/vite.config.ts`).

---

## Usage
1. Start the backend (`uvicorn`) and the frontend (`npm run dev`), then open **http://localhost:5173**.
2. Upload a **master** (golden reference) image and a **live** specimen image.
3. Optionally adjust the rotation control to roughly orient the live piece.
4. Click **Run Verification**. The live image passes through all eight stages.
5. Read the verdict (ACCEPT / REVIEW / REJECT), inspect the per-check heatmaps and bounding boxes, and review the per-stage timing.

For consistent results, capture master and live images under the same lighting and camera setup against a matte black or white background, and save as lossless PNG (JPEG compression artifacts can mask real defects). See [jewelry_inspection_validation_plan.md](jewelry_inspection_validation_plan.md) for the full dataset and calibration procedure.

---

## Project Structure
```
Jewelry-Visual-Inspection/
│
├── backend/                         # FastAPI service + 8-stage pipeline
│   ├── main.py                      # API endpoints (/api/info, /api/inspect, /api/healthz)
│   └── pipeline/
│       ├── preprocess.py            # Stage 1 — load & resize to working resolution
│       ├── segmentation.py          # Stage 2 — mask, bbox, centroid
│       ├── rotation.py              # Stage 3 — moment-based similarity alignment
│       ├── registration.py          # Stage 4 — SIFT + RANSAC homography
│       ├── profile_check.py         # Stage 5 — Hu moments, edge IoU, area deviation
│       ├── decoration_check.py      # Stage 6 — DINOv2 patch similarity + LAB ΔE
│       ├── surface_check.py         # Stage 7 — SSIM + Laplacian + color anomaly
│       ├── decision.py              # Stage 8 — ACCEPT / REVIEW / REJECT
│       ├── orchestrator.py          # Runs all stages with per-stage timing
│       ├── visualizer.py            # Edge overlays, heatmaps, bounding boxes
│       └── types.py                 # Shared dataclasses
│
├── frontend/                        # Vite + React + TypeScript UI
│   ├── index.html
│   ├── vite.config.ts               # Dev server + /api proxy
│   └── src/
│       ├── App.tsx, main.tsx
│       ├── api.ts, types.ts
│       ├── components/              # Masthead, Hero, UploadRow, Verdict,
│       │                           # VisualDisplay, CheckResults, TimingRow, ...
│       ├── utils/rotate.ts
│       └── styles/app.css
│
├── Images/                          # Sample master / defect test images
├── requirements.txt                 # Backend Python dependencies
├── jewelry_inspection_validation_plan.md   # Dataset + accuracy/latency validation plan
└── README.md                        # Project documentation
```

---

## Validation & Acceptance
From the validation plan, the system targets:
- **Speed:** total pipeline **P95 < 1000 ms**
- **Accuracy:** **0% false accept** on the test set (a defective piece must never be auto-accepted)
- **Localization:** **defect-localization recall > 90%**

Borderline pieces should route to **REVIEW** rather than being auto-accepted, which is the safety margin that keeps false accepts at zero.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com
