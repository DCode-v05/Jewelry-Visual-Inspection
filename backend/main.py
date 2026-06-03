"""FastAPI service wrapping the 8-stage jewelry inspection pipeline."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .pipeline import orchestrator
from .pipeline.decoration_check import DEFAULT_MODEL_NAME, init_model
from .pipeline.types import BoundingBox, PipelineResult
from .pipeline import visualizer


app = FastAPI(title="Jewelry Visual Inspection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- response models ----------


class BoundingBoxModel(BaseModel):
    x: int
    y: int
    w: int
    h: int
    color: str
    label: str
    score: Optional[float] = None
    points: Optional[List[List[int]]] = None  # rotated/polygon corners, optional


class CheckResultModel(BaseModel):
    name: str
    verdict: str
    metrics: dict
    boxes: List[BoundingBoxModel]
    heatmap_png: Optional[str] = None


class StageTimingModel(BaseModel):
    name: str
    ms: float


class InspectResponse(BaseModel):
    decision: str
    reasons: List[str]
    rotation_deg: float
    registration: dict
    profile: CheckResultModel
    decoration: CheckResultModel
    surface: CheckResultModel
    timings: List[StageTimingModel]
    total_ms: float
    # Master cell
    master_png: str
    master_contoured_png: str
    # Live cell — horizontal stage progression
    live_preprocessed_png: str
    live_segmented_png: str
    live_rotated_png: str
    live_registered_png: str


class InfoResponse(BaseModel):
    architecture: str
    model_name: str
    device: str
    working_resolution: int
    pipeline_stages: List[str]
    bbox_colors: dict


# ---------- helpers ----------


def _png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _boxes_to_models(boxes: List[BoundingBox]) -> List[BoundingBoxModel]:
    return [BoundingBoxModel(**b.__dict__) for b in boxes]


def _serialize(result: PipelineResult) -> InspectResponse:
    surface_heatmap_bgr = visualizer.colorize_surface_heatmap(result.surface.defect_map)

    profile = CheckResultModel(
        name="profile",
        verdict=result.profile.verdict,
        metrics={
            "edge_iou": round(result.profile.silhouette_iou, 5),
            "missing_edge_ratio": round(result.profile.missing_edge_ratio, 5),
            "area_deviation": round(result.profile.area_deviation, 5),
            "shape_distance": round(result.profile.shape_distance, 5),
        },
        boxes=_boxes_to_models(result.profile.boxes),
        heatmap_png=_png_b64(result.profile.diff_overlay),
    )

    decoration = CheckResultModel(
        name="decoration",
        verdict=result.decoration.verdict,
        metrics={
            "global_similarity": round(result.decoration.global_similarity, 5),
            "problem_patch_ratio": round(result.decoration.problem_patch_ratio, 5),
            "max_color_distance": round(result.decoration.max_color_distance, 2),
        },
        boxes=_boxes_to_models(result.decoration.boxes),
        heatmap_png=_png_b64(result.decoration.diff_overlay),
    )

    surface = CheckResultModel(
        name="surface",
        verdict=result.surface.verdict,
        metrics={
            "defect_ratio": round(result.surface.defect_ratio, 5),
            "num_defect_regions": result.surface.num_defect_regions,
            "max_defect_size": result.surface.max_defect_size,
            "max_color_distance": round(result.surface.max_color_distance, 2),
            "num_scratches": result.surface.num_scratches,
            "num_stone_defects": result.surface.num_stone_defects,
        },
        boxes=_boxes_to_models(result.surface.boxes),
        heatmap_png=_png_b64(surface_heatmap_bgr),
    )

    return InspectResponse(
        decision=result.decision,
        reasons=result.reasons,
        rotation_deg=round(result.rotation.angle_deg, 3),
        registration={
            "reliable": result.registration.reliable,
            "ncc": round(result.registration.ncc, 5),
            "num_inliers": result.registration.num_inliers,
            "inlier_ratio": round(result.registration.inlier_ratio, 5),
        },
        profile=profile,
        decoration=decoration,
        surface=surface,
        timings=[StageTimingModel(name=t.name, ms=round(t.ms, 3)) for t in result.timings],
        total_ms=round(result.total_ms, 3),
        master_png=_png_b64(result.master_image),
        master_contoured_png=_png_b64(result.master_contoured),
        live_preprocessed_png=_png_b64(result.live_preprocessed),
        live_segmented_png=_png_b64(result.live_segmented),
        live_rotated_png=_png_b64(result.live_rotated),
        live_registered_png=_png_b64(result.live_registered),
    )


# ---------- endpoints ----------


@app.on_event("startup")
async def _startup() -> None:
    init_model()
    try:
        orchestrator.warmup(2)
    except Exception:
        # Warmup is best-effort; first real request will pay the cost.
        pass


@app.get("/api/healthz")
async def healthz() -> dict:
    return {"ok": True}


@app.get("/api/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return InfoResponse(
        architecture="DinoV2 patch + SIFT + SSIM",
        model_name=DEFAULT_MODEL_NAME,
        device=device,
        working_resolution=1024,
        pipeline_stages=[
            "preprocess",
            "segmentation",
            "rotation_estimation",
            "fine_registration",
            "profile_check",
            "decoration_check",
            "surface_check",
            "decision",
        ],
        bbox_colors={
            "profile": "#FF0000",
            "decoration": "#FFA500",
            "surface": "#FF00FF",
            "master_contour": "#00FF00",
            "live_contour": "#FFFF00",
        },
    )


@app.post("/api/inspect", response_model=InspectResponse)
async def inspect(
    master: UploadFile = File(...),
    live: UploadFile = File(...),
) -> InspectResponse:
    master_bytes = await master.read()
    live_bytes = await live.read()
    if not master_bytes or not live_bytes:
        return JSONResponse(status_code=400, content={"error": "both images required"})
    result = orchestrator.run_pipeline(master_bytes, live_bytes)
    return _serialize(result)
