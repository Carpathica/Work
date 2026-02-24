from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import Box, PredictRequest, SaveAllRequest, SaveAnnotationsRequest, SessionRequest
from .storage import (
    discover_images,
    image_absolute_path,
    load_classes,
    read_annotations,
    resolve_path,
    save_annotations,
)
from .yolo_inference import YoloRunner


@dataclass
class SessionState:
    dataset_dir: Path | None = None
    images: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    model_path: Path | None = None
    labels_dir: str | None = None


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="YOLO Web Annotator", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_state = SessionState()
_lock = Lock()
_runner = YoloRunner()


def _state_snapshot() -> SessionState:
    with _lock:
        return SessionState(
            dataset_dir=_state.dataset_dir,
            images=list(_state.images),
            classes=list(_state.classes),
            model_path=_state.model_path,
            labels_dir=_state.labels_dir,
        )


def _require_dataset() -> SessionState:
    snapshot = _state_snapshot()
    if snapshot.dataset_dir is None:
        raise HTTPException(status_code=400, detail="Session is not configured. Call POST /api/session first.")
    return snapshot


def _ensure_image_known(snapshot: SessionState, image_path: str) -> None:
    if image_path not in snapshot.images:
        raise HTTPException(status_code=404, detail=f"Image is not in the active dataset: {image_path}")


def _session_payload(snapshot: SessionState) -> dict:
    return {
        "dataset_dir": str(snapshot.dataset_dir) if snapshot.dataset_dir else None,
        "image_count": len(snapshot.images),
        "images": snapshot.images,
        "classes": snapshot.classes,
        "model_path": str(snapshot.model_path) if snapshot.model_path else None,
        "labels_dir": snapshot.labels_dir,
    }


def _list_roots() -> List[str]:
    if os.name == "nt":
        roots: List[str] = []
        for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = Path(f"{drive}:\\")
            if root.exists():
                roots.append(str(root))
        return roots
    return ["/"]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/session")
def get_session() -> dict:
    return _session_payload(_state_snapshot())


@app.post("/api/session")
def set_session(payload: SessionRequest) -> dict:
    dataset_dir = resolve_path(Path.cwd(), payload.dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Dataset directory does not exist: {dataset_dir}")

    classes = load_classes(dataset_dir, payload.classes, payload.classes_file)
    images = discover_images(dataset_dir)
    if not images:
        raise HTTPException(status_code=400, detail="No images found in the selected dataset directory.")

    model_path = None
    if payload.model_path:
        model_path = resolve_path(dataset_dir, payload.model_path)

    labels_dir = None
    if payload.labels_dir:
        labels_root = resolve_path(dataset_dir, payload.labels_dir)
        if labels_root.exists() and not labels_root.is_dir():
            raise HTTPException(status_code=400, detail=f"Labels path is not a directory: {labels_root}")
        labels_dir = str(labels_root)

    with _lock:
        _state.dataset_dir = dataset_dir
        _state.images = images
        _state.classes = classes
        _state.model_path = model_path
        _state.labels_dir = labels_dir

    return _session_payload(_state_snapshot())


@app.get("/api/images")
def get_images() -> dict:
    snapshot = _require_dataset()
    return {"images": snapshot.images}


@app.get("/api/image")
def get_image(path: str = Query(..., description="Relative image path inside dataset")) -> FileResponse:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    image_path = image_absolute_path(snapshot.dataset_dir, path)  # type: ignore[arg-type]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found: {path}")
    return FileResponse(image_path)


@app.get("/api/annotations")
def get_annotations(path: str = Query(..., description="Relative image path inside dataset")) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    try:
        boxes = read_annotations(snapshot.dataset_dir, path, labels_dir=snapshot.labels_dir)  # type: ignore[arg-type]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"boxes": [box.model_dump() for box in boxes]}


@app.post("/api/annotations")
def put_annotations(
    payload: SaveAnnotationsRequest,
    path: str = Query(..., description="Relative image path inside dataset"),
) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    try:
        output_path = save_annotations(
            snapshot.dataset_dir,  # type: ignore[arg-type]
            path,
            payload.boxes,
            labels_dir=snapshot.labels_dir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"saved": True, "label_path": str(output_path)}


@app.post("/api/annotations/batch")
def put_annotations_batch(payload: SaveAllRequest) -> dict:
    snapshot = _require_dataset()
    saved_count = 0
    errors: List[dict] = []

    for item in payload.items:
        try:
            _ensure_image_known(snapshot, item.path)
            save_annotations(
                snapshot.dataset_dir,  # type: ignore[arg-type]
                item.path,
                item.boxes,
                labels_dir=snapshot.labels_dir,
            )
            saved_count += 1
        except (HTTPException, FileNotFoundError, ValueError) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            errors.append({"path": item.path, "error": detail})

    return {"saved_count": saved_count, "error_count": len(errors), "errors": errors}


@app.post("/api/predict")
def predict(
    payload: PredictRequest,
    path: str = Query(..., description="Relative image path inside dataset"),
) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    image_path = image_absolute_path(snapshot.dataset_dir, path)  # type: ignore[arg-type]

    if payload.model_path:
        model_path = resolve_path(snapshot.dataset_dir, payload.model_path)  # type: ignore[arg-type]
    elif snapshot.model_path:
        model_path = snapshot.model_path
    else:
        raise HTTPException(status_code=400, detail="Model path is not set. Provide it in session or predict request.")

    try:
        predicted_boxes, model_classes = _runner.predict(
            image_path=image_path,
            model_path=model_path,
            conf=payload.conf,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    class_list = list(snapshot.classes)
    if not class_list and model_classes:
        max_index = max(model_classes.keys())
        class_list = [model_classes.get(i, str(i)) for i in range(max_index + 1)]
        with _lock:
            _state.classes = class_list

    return {
        "boxes": [box.model_dump() for box in predicted_boxes],
        "model_classes": model_classes,
        "classes": class_list,
    }


@app.delete("/api/annotations")
def clear_annotations(path: str = Query(..., description="Relative image path inside dataset")) -> dict:
    snapshot = _require_dataset()
    _ensure_image_known(snapshot, path)
    empty: List[Box] = []
    output_path = save_annotations(
        snapshot.dataset_dir,  # type: ignore[arg-type]
        path,
        empty,
        labels_dir=snapshot.labels_dir,
    )
    return {"saved": True, "label_path": str(output_path)}


@app.get("/api/fs/roots")
def get_fs_roots() -> dict:
    return {"roots": _list_roots()}


@app.get("/api/fs/list")
def list_fs(
    path: str | None = Query(default=None, description="Directory path to inspect"),
    mode: str = Query(default="all", description="all|dir|model|yaml"),
) -> dict:
    if mode not in {"all", "dir", "model", "yaml"}:
        raise HTTPException(status_code=400, detail="Unsupported mode. Use all, dir, model or yaml.")

    current = resolve_path(Path.cwd(), path) if path else Path.cwd().resolve()
    if not current.exists() or not current.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {current}")

    try:
        entries = list(current.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"Permission denied: {current}") from exc

    directories = sorted([entry for entry in entries if entry.is_dir()], key=lambda item: item.name.lower())
    files = sorted([entry for entry in entries if entry.is_file()], key=lambda item: item.name.lower())

    if mode == "model":
        files = [entry for entry in files if entry.suffix.lower() in {".pt", ".onnx"}]
    elif mode == "yaml":
        files = [entry for entry in files if entry.suffix.lower() in {".yaml", ".yml", ".txt"}]
    elif mode == "dir":
        files = []

    parent = current.parent
    has_parent = parent != current

    return {
        "current_path": str(current),
        "parent_path": str(parent) if has_parent else None,
        "directories": [{"name": entry.name, "path": str(entry)} for entry in directories],
        "files": [{"name": entry.name, "path": str(entry)} for entry in files],
    }
