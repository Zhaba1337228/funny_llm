from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.core.config import get_settings
from backend.app.schemas.api import PredictionRequest, TrainingRequest
from backend.app.services.dataset_service import DatasetService
from backend.app.services.eda_service import EDAService
from backend.app.services.system_service import SystemService
from backend.app.services.training_service import TrainingService


settings = get_settings()
dataset_service = DatasetService(settings)
eda_service = EDAService(dataset_service, settings)
system_service = SystemService()
training_service = TrainingService(dataset_service, settings)

app = FastAPI(title=settings.app_name, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=settings.allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def handle_error(exc: Exception, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(exc))


@app.get("/api/health")
def healthcheck() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.get("/api/dataset/info")
def dataset_info() -> dict:
    try:
        return dataset_service.get_dashboard_snapshot()
    except Exception as exc:
        raise handle_error(exc, 500) from exc


@app.get("/api/dataset/preview")
def dataset_preview(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=settings.preview_page_size, ge=5, le=200),
    search: str | None = None,
    sort_by: str | None = None,
    sort_dir: str = Query(default="desc", pattern="^(asc|desc)$"),
) -> dict:
    try:
        return dataset_service.get_preview(page=page, page_size=page_size, search=search, sort_by=sort_by, sort_dir=sort_dir)
    except Exception as exc:
        raise handle_error(exc, 500) from exc


@app.get("/api/eda/summary")
def eda_summary() -> dict:
    try:
        return eda_service.get_summary()
    except Exception as exc:
        raise handle_error(exc, 500) from exc


@app.post("/api/train/start")
def train_start(request: TrainingRequest) -> dict:
    try:
        return training_service.start_training(request)
    except Exception as exc:
        raise handle_error(exc) from exc


@app.post("/api/train/stop")
def train_stop() -> dict:
    try:
        return training_service.stop_training()
    except Exception as exc:
        raise handle_error(exc) from exc


@app.get("/api/train/status")
def train_status() -> dict:
    return training_service.get_status()


@app.websocket("/api/ws/training-status")
async def train_status_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    last_payload: dict | None = None
    try:
        while True:
            payload = training_service.get_status()
            if payload != last_payload:
                await websocket.send_json(payload)
                last_payload = payload
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/api/train/results")
def train_results() -> dict:
    try:
        return training_service.get_results()
    except Exception as exc:
        raise handle_error(exc, 404) from exc


@app.get("/api/models/list")
def models_list() -> dict:
    try:
        return training_service.list_models()
    except Exception as exc:
        raise handle_error(exc, 500) from exc


@app.post("/api/models/select/{run_id}")
def models_select(run_id: str) -> dict:
    try:
        return training_service.select_model(run_id)
    except FileNotFoundError as exc:
        raise handle_error(exc, 404) from exc
    except Exception as exc:
        raise handle_error(exc) from exc


@app.get("/api/models/compare")
def models_compare() -> dict:
    return training_service.get_comparison()


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        return training_service.predict(request.features)
    except Exception as exc:
        raise handle_error(exc) from exc


@app.get("/api/candidates/top")
def candidates_top(
    limit: int = Query(default=50, ge=1, le=500),
    search: str | None = None,
    min_score: float | None = Query(default=None, ge=0, le=100),
    experience_min: float | None = Query(default=None, ge=0),
    internships_min: int | None = Query(default=None, ge=0),
    certifications_min: int | None = Query(default=None, ge=0),
    skill_score_min: float | None = Query(default=None, ge=0, le=100),
    company_type: str | None = None,
) -> dict:
    try:
        return training_service.get_top_candidates(
            limit=limit,
            search=search,
            min_score=min_score,
            experience_min=experience_min,
            internships_min=internships_min,
            certifications_min=certifications_min,
            skill_score_min=skill_score_min,
            company_type=company_type,
        )
    except Exception as exc:
        raise handle_error(exc, 404) from exc


@app.get("/api/candidates/{candidate_id}")
def candidate_detail(candidate_id: int) -> dict:
    try:
        return training_service.get_candidate_detail(candidate_id)
    except KeyError as exc:
        raise handle_error(exc, 404) from exc
    except Exception as exc:
        raise handle_error(exc) from exc


@app.get("/api/system/device")
def system_device() -> dict:
    return system_service.get_device_info()


@app.get("/api/candidates/export")
def export_candidates() -> FileResponse:
    try:
        export_path = training_service.export_ranking()
        return FileResponse(path=str(export_path), filename=export_path.name, media_type="text/csv")
    except Exception as exc:
        raise handle_error(exc, 404) from exc
