#!/usr/bin/env python3
"""
Collect two-camera Axis snapshots for YOLO training.

The script fetches a pair of images, runs the existing recogniser/read_container.py
logic on each image, and saves the pair into accepted/manual YOLO-style folders.
It intentionally does not modify recogniser/read_container.py.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import cv2
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
RECOGNISER_DIR = REPO_ROOT / "recogniser"
if str(RECOGNISER_DIR) not in sys.path:
    sys.path.insert(0, str(RECOGNISER_DIR))

import read_container  # noqa: E402


ACTIVE_RECOGNIZER_MODULE = read_container


DEFAULT_WEIGHTS = "runs/detect/model_8_05_26_278e_26m/best.pt"
DEFAULT_OUTPUT_ROOT = "datasets/axis_collected"
DEFAULT_STATE_FILE = "runs/axis_dual_collector_state.json"
DEFAULT_LOG_JSONL = "runs/axis_dual_collector.jsonl"
COLLECTOR_MODE_DUAL = "dual_camera"
COLLECTOR_MODE_SINGLE = "single_camera"
COLLECTOR_MODES = {COLLECTOR_MODE_DUAL, COLLECTOR_MODE_SINGLE}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CameraConfig:
    name: str
    host: str | None
    path: str
    url: str | None
    user: str | None
    password_env: str | None
    password: str | None
    auth: str
    timeout_seconds: float


@dataclass(frozen=True)
class Capture:
    camera: CameraConfig
    path: Path
    source_name: str
    suffix: str


@dataclass(frozen=True)
class Analysis:
    capture: Capture
    read: Any

    @property
    def primary_number(self) -> str:
        return normalize_number(self.read.primary_number)

    @property
    def detections_count(self) -> int:
        return len(self.read.ordered)


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    number: str | None
    category: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect accepted/manual two-camera container OCR samples."
    )
    parser.add_argument("--config", required=True, type=Path, help="JSON collector config.")
    parser.add_argument("--source", type=Path, default=None, help="Local test image for single-camera mode.")
    parser.add_argument("--source1", type=Path, default=None, help="Local test image for camera 1.")
    parser.add_argument("--source2", type=Path, default=None, help="Local test image for camera 2.")
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit.")
    parser.add_argument("--output-root", type=Path, default=None, help="Override config output_root.")
    parser.add_argument("--state-file", type=Path, default=None, help="Override config state_file.")
    parser.add_argument("--log-jsonl", type=Path, default=None, help="Override config log_jsonl.")
    parser.add_argument("--min-detections", type=int, default=None, help="Override config min_detections.")
    parser.add_argument(
        "--recognizer-module",
        default=None,
        help="Recognizer module name or .py path. Defaults to config recognizer_module or read_container.",
    )
    parser.add_argument("--interval-minutes", type=float, default=None, help="Override polling interval.")
    parser.add_argument(
        "--no-single-camera-on-error",
        action="store_true",
        help="Disable saving a useful single-camera manual sample when the other live camera fails.",
    )
    parser.add_argument(
        "--require-both-detections",
        action="store_true",
        help="Save a two-camera sample only when both images have at least min_detections.",
    )
    return parser.parse_args()


def load_recognizer_module(value: str | Path | None):
    raw = str(value or "read_container")
    raw_path = Path(raw)
    is_path = raw_path.suffix == ".py" or any(sep in raw for sep in ("/", "\\"))

    if not is_path:
        return importlib.import_module(raw)

    module_path = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Recognizer module file not found: {module_path}")

    module_name = module_path.stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load recognizer module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def recognizer_default_merge_iou(module) -> float:
    return float(getattr(module, "DEFAULT_MERGE_IOU", read_container.DEFAULT_MERGE_IOU))


def predict_with_recognizer(
    module,
    model: YOLO,
    image_path: Path,
    *,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: float | None,
):
    if hasattr(module, "predict_camera_read"):
        return module.predict_camera_read(
            model,
            image_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            merge_iou=merge_iou,
        )

    if hasattr(module, "predict_container_kp_with_layout"):
        text, check_ok, ordered, layout, size_type = module.predict_container_kp_with_layout(
            model,
            image_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            merge_iou=merge_iou,
        )
        return SimpleNamespace(
            image_path=image_path,
            primary_number=text,
            check_ok=check_ok,
            ordered=ordered,
            size_type_code=size_type,
            layout=layout,
            char_scores=[],
        )

    if hasattr(module, "predict_container_read"):
        read = module.predict_container_read(
            model,
            image_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            merge_iou=merge_iou,
        )
        return SimpleNamespace(
            image_path=image_path,
            primary_number=read.primary_number,
            check_ok=read.check_ok,
            ordered=read.ordered,
            size_type_code=read.size_type_code,
            layout=read.layout,
            char_scores=getattr(read, "char_scores", []),
        )

    raise AttributeError(
        f"Recognizer module {module.__name__!r} must define predict_camera_read(), "
        "predict_container_kp_with_layout(), or predict_container_read()"
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return data


def resolve_repo_path(value: str | Path | None, default: str) -> Path:
    raw = Path(value or default)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def parse_collector_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("collector_mode") or COLLECTOR_MODE_DUAL).lower()
    if mode not in COLLECTOR_MODES:
        raise ValueError(
            f"Unsupported collector_mode: {mode!r}. "
            f"Use one of: {', '.join(sorted(COLLECTOR_MODES))}"
        )
    return mode


def parse_camera_configs(config: dict[str, Any], collector_mode: str) -> list[CameraConfig]:
    raw_cameras = config.get("cameras")
    expected_count = 1 if collector_mode == COLLECTOR_MODE_SINGLE else 2
    if not isinstance(raw_cameras, list) or len(raw_cameras) != expected_count:
        raise ValueError(
            f"Config with collector_mode={collector_mode!r} must contain "
            f"exactly {expected_count} camera(s)"
        )

    cameras: list[CameraConfig] = []
    for idx, raw in enumerate(raw_cameras, start=1):
        if not isinstance(raw, dict):
            raise ValueError("Each camera config must be an object")
        cameras.append(
            CameraConfig(
                name=str(raw.get("name") or f"camera{idx}"),
                host=str(raw["host"]) if raw.get("host") else None,
                path=str(raw.get("path") or "/axis-cgi/jpg/image.cgi"),
                url=str(raw["url"]) if raw.get("url") else None,
                user=str(raw["user"]) if raw.get("user") else None,
                password_env=str(raw["password_env"]) if raw.get("password_env") else None,
                password=str(raw["password"]) if raw.get("password") else None,
                auth=str(raw.get("auth") or "auto").lower(),
                timeout_seconds=float(raw.get("timeout_seconds", 15.0)),
            )
        )

    for cam in cameras:
        if cam.auth not in {"auto", "basic", "digest", "none"}:
            raise ValueError(f"Unsupported auth mode for {cam.name}: {cam.auth}")
    return cameras


def build_snapshot_url(camera: CameraConfig) -> str:
    if camera.url:
        return camera.url
    if not camera.host:
        raise ValueError(f"Camera {camera.name} needs either url or host")

    host = camera.host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    path = camera.path if camera.path.startswith("/") else f"/{camera.path}"
    return f"{host}{path}"


def camera_password(camera: CameraConfig) -> str | None:
    if camera.password is not None:
        return camera.password
    if camera.password_env:
        return os.environ.get(camera.password_env)
    return None


def looks_like_image(content: bytes) -> bool:
    return (
        content.startswith(b"\xff\xd8")  # JPEG
        or content.startswith(b"\x89PNG\r\n\x1a\n")
        or content.startswith(b"BM")  # BMP
        or content.startswith((b"II*\x00", b"MM\x00*"))  # TIFF
        or (len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP")
    )


def request_snapshot(camera: CameraConfig) -> bytes:
    url = build_snapshot_url(camera)
    user = camera.user
    password = camera_password(camera) or ""

    if not user or camera.auth == "none":
        auth_candidates: list[object | None] = [None]
    elif camera.auth == "basic":
        auth_candidates = [HTTPBasicAuth(user, password)]
    elif camera.auth == "digest":
        auth_candidates = [HTTPDigestAuth(user, password)]
    else:
        auth_candidates = [HTTPDigestAuth(user, password), HTTPBasicAuth(user, password)]

    last_error: Exception | None = None
    for idx, auth in enumerate(auth_candidates):
        try:
            response = requests.get(url, auth=auth, timeout=camera.timeout_seconds)
            if response.status_code == 401 and idx < len(auth_candidates) - 1:
                continue
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "image" not in content_type and not looks_like_image(response.content):
                raise ValueError(
                    f"{camera.name} response is not an image: content-type={content_type!r}"
                )
            return response.content
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No request attempted for {camera.name}")


def normalize_number(value: str | None) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def is_complete_number(value: str) -> bool:
    return len(value) == 11 and value[:4].isalpha() and value[4:].isdigit()


def manual_duplicate_key(analyses: Sequence[Analysis]) -> str:
    complete_numbers = sorted(
        {
            analysis.primary_number
            for analysis in analyses
            if is_complete_number(analysis.primary_number)
        }
    )
    if complete_numbers:
        return "complete:" + "|".join(complete_numbers)

    camera_numbers = [
        f"{analysis.capture.camera.name}:{analysis.primary_number or '-'}"
        for analysis in analyses
    ]
    return "raw:" + "|".join(camera_numbers)


def has_check_ok_number(analysis: Analysis) -> bool:
    return bool(analysis.read.check_ok)


def manual_category(analyses: Sequence[Analysis]) -> str:
    if any(has_check_ok_number(analysis) for analysis in analyses):
        return "manual/recognized"
    return "manual/unrecognized"


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_state(path: Path, updates: dict[str, Any]) -> None:
    state = load_state(path)
    state.update(updates)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(path, state)


def append_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_recognition_info(
    *,
    output_root: Path,
    category: str,
    live_timestamp: datetime | None,
    analyses: Sequence[Analysis],
    saved: Sequence[dict[str, str]],
) -> None:
    if category != "manual/recognized":
        return

    date_text = (live_timestamp or datetime.now()).date().isoformat()
    info_path = output_root / category / date_text / "recognition_info.jsonl"
    recognized = [
        {
            "camera": analysis.capture.camera.name,
            "primary_number": analysis.primary_number,
            "check_ok": bool(analysis.read.check_ok),
            "detections_count": analysis.detections_count,
            "saved_image": next(
                (item["image"] for item in saved if item["camera"] == analysis.capture.camera.name),
                None,
            ),
        }
        for analysis in analyses
        if has_check_ok_number(analysis)
    ]
    append_log(
        info_path,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "recognized": recognized,
            "all_cameras": [analysis_record(analysis) for analysis in analyses],
        },
    )


def build_class_id_map(model: YOLO) -> dict[str, int]:
    names = model.names if hasattr(model, "names") else {}
    if isinstance(names, dict):
        return {str(label).upper(): int(idx) for idx, label in names.items()}
    return {str(label).upper(): idx for idx, label in enumerate(names)}


def yolo_label_lines(
    analysis: Analysis,
    class_ids: dict[str, int],
    image_path: Path,
) -> list[str]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read saved image for labels: {image_path}")

    height, width = image.shape[:2]
    lines: list[str] = []
    for det in analysis.read.ordered:
        label = normalize_number(det.label)
        if len(label) != 1 or label not in class_ids:
            continue

        x1, y1, x2, y2 = det.xyxy
        x1 = max(0.0, min(float(width), float(x1)))
        y1 = max(0.0, min(float(height), float(y1)))
        x2 = max(0.0, min(float(width), float(x2)))
        y2 = max(0.0, min(float(height), float(y2)))
        if x2 <= x1 or y2 <= y1:
            continue

        xc = ((x1 + x2) * 0.5) / width
        yc = ((y1 + y2) * 0.5) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{class_ids[label]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def write_label_file(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")


def unique_output_paths(
    images_dir: Path,
    labels_dir: Path,
    stem: str,
    suffix: str,
) -> tuple[Path, Path]:
    suffix = suffix if suffix.lower() in IMAGE_SUFFIXES else ".jpg"
    candidate = stem
    counter = 2
    while True:
        image_path = images_dir / f"{candidate}{suffix}"
        label_path = labels_dir / f"{candidate}.txt"
        if not image_path.exists() and not label_path.exists():
            return image_path, label_path
        candidate = f"{stem}_{counter}"
        counter += 1


def make_stems(captures: Sequence[Capture], *, live_timestamp: datetime | None) -> list[str]:
    if live_timestamp is not None:
        time_part = live_timestamp.strftime("%H-%M-%S")
        return [f"{capture.camera.name}_{time_part}" for capture in captures]

    stems = [capture.path.stem for capture in captures]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{capture.path.stem}_{capture.camera.name}" for capture in captures]


def save_pair(
    analyses: Sequence[Analysis],
    *,
    category: str,
    output_root: Path,
    class_ids: dict[str, int],
    live_timestamp: datetime | None,
) -> list[dict[str, str]]:
    date_text = (live_timestamp or datetime.now()).date().isoformat()
    images_dir = output_root / category / date_text / "images"
    labels_dir = output_root / category / date_text / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    captures = [analysis.capture for analysis in analyses]
    stems = make_stems(captures, live_timestamp=live_timestamp)
    saved: list[dict[str, str]] = []
    for analysis, stem in zip(analyses, stems):
        image_path, label_path = unique_output_paths(
            images_dir,
            labels_dir,
            stem,
            analysis.capture.suffix,
        )
        shutil.copyfile(analysis.capture.path, image_path)
        write_label_file(label_path, yolo_label_lines(analysis, class_ids, image_path))
        saved.append(
            {
                "camera": analysis.capture.camera.name,
                "image": str(image_path),
                "label": str(label_path),
            }
        )
    return saved


def decide(
    analyses: Sequence[Analysis],
    *,
    min_detections: int,
    last_number: str | None,
    last_manual_key: str | None,
    require_both_detections: bool,
) -> Decision:
    detections_ok = [analysis.detections_count >= min_detections for analysis in analyses]
    has_container = all(detections_ok) if require_both_detections else any(detections_ok)
    if not has_container:
        reason = "not_enough_detections_on_both_images" if require_both_detections else "not_enough_detections"
        return Decision("skip", reason, None, None)

    numbers = [analysis.primary_number for analysis in analyses]
    both_complete = all(is_complete_number(number) for number in numbers)
    both_verified = all(has_check_ok_number(analysis) for analysis in analyses)
    if both_complete and both_verified and numbers[0] == numbers[1]:
        if last_number and numbers[0] == last_number:
            return Decision("skip", "duplicate_accepted_number", numbers[0], None)
        return Decision("save", "accepted_number_match", numbers[0], "accepted")

    manual_key = manual_duplicate_key(analyses)
    if last_manual_key and manual_key == last_manual_key:
        return Decision("skip", "duplicate_manual_case", None, None)

    return Decision("save", "manual_number_missing_or_mismatch", None, manual_category(analyses))


def decide_single_camera(
    analysis: Analysis,
    *,
    min_detections: int,
    last_number: str | None,
    last_manual_key: str | None,
) -> Decision:
    if analysis.detections_count < min_detections:
        return Decision("skip", "not_enough_detections", None, None)

    number = analysis.primary_number
    if is_complete_number(number) and has_check_ok_number(analysis):
        if last_number and number == last_number:
            return Decision("skip", "duplicate_accepted_number", number, None)
        return Decision("save", "accepted_single_camera_number", number, "accepted")

    manual_key = manual_duplicate_key([analysis])
    if last_manual_key and manual_key == last_manual_key:
        return Decision("skip", "duplicate_manual_case", None, None)

    return Decision("save", "manual_single_camera_number_missing", None, manual_category([analysis]))


def analyse_captures(
    captures: Sequence[Capture],
    *,
    model: YOLO,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: float | None,
) -> list[Analysis]:
    out: list[Analysis] = []
    for capture in captures:
        read = predict_with_recognizer(
            ACTIVE_RECOGNIZER_MODULE,
            model,
            capture.path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            merge_iou=merge_iou,
        )
        out.append(Analysis(capture, read))
    return out


def local_captures(cameras: Sequence[CameraConfig], source1: Path, source2: Path) -> list[Capture]:
    sources = [source1, source2]
    captures: list[Capture] = []
    for camera, source in zip(cameras, sources):
        path = source.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        captures.append(
            Capture(
                camera=camera,
                path=path,
                source_name=path.name,
                suffix=path.suffix if path.suffix else ".jpg",
            )
        )
    return captures


def local_single_capture(camera: CameraConfig, source: Path) -> list[Capture]:
    path = source.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        Capture(
            camera=camera,
            path=path,
            source_name=path.name,
            suffix=path.suffix if path.suffix else ".jpg",
        )
    ]


def live_captures(cameras: Sequence[CameraConfig], temp_dir: Path) -> list[Capture]:
    captures: list[Capture] = []
    for camera in cameras:
        snapshot = request_snapshot(camera)
        path = temp_dir / f"{camera.name}.jpg"
        path.write_bytes(snapshot)
        captures.append(Capture(camera=camera, path=path, source_name=path.name, suffix=".jpg"))
    return captures


def live_captures_partial(
    cameras: Sequence[CameraConfig],
    temp_dir: Path,
) -> tuple[list[Capture], list[dict[str, str]]]:
    captures: list[Capture] = []
    errors: list[dict[str, str]] = []
    for camera in cameras:
        try:
            snapshot = request_snapshot(camera)
            path = temp_dir / f"{camera.name}.jpg"
            path.write_bytes(snapshot)
            captures.append(Capture(camera=camera, path=path, source_name=path.name, suffix=".jpg"))
        except Exception as exc:
            errors.append({"camera": camera.name, "error": str(exc)})
    return captures, errors


def analysis_record(analysis: Analysis) -> dict[str, Any]:
    return {
        "camera": analysis.capture.camera.name,
        "image": str(analysis.capture.path),
        "primary_number": analysis.primary_number,
        "check_ok": bool(analysis.read.check_ok),
        "detections_count": analysis.detections_count,
        "size_type_code": analysis.read.size_type_code,
        "layout": analysis.read.layout,
    }


def print_iteration(decision: Decision, analyses: Sequence[Analysis]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    camera_text = " | ".join(
        (
            f"{analysis.capture.camera.name}: "
            f"num={analysis.primary_number or '-'} "
            f"dets={analysis.detections_count} "
            f"ok={analysis.read.check_ok}"
        )
        for analysis in analyses
    )
    print(
        f"{timestamp} {decision.action.upper()} {decision.reason} "
        f"category={decision.category or '-'} number={decision.number or '-'} :: {camera_text}",
        flush=True,
    )


def process_iteration(
    captures: Sequence[Capture],
    *,
    model: YOLO,
    class_ids: dict[str, int],
    output_root: Path,
    state_path: Path,
    log_path: Path,
    min_detections: int,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: float | None,
    require_both_detections: bool,
    mode: str,
    live_timestamp: datetime | None,
) -> Decision:
    state = load_state(state_path)
    analyses = analyse_captures(
        captures,
        model=model,
        conf=conf,
        iou=iou,
        max_det=max_det,
        merge_iou=merge_iou,
    )
    decision = decide(
        analyses,
        min_detections=min_detections,
        last_number=normalize_number(state.get("last_accepted_number")),
        last_manual_key=str(state.get("last_manual_key") or ""),
        require_both_detections=require_both_detections,
    )

    saved: list[dict[str, str]] = []
    if decision.action == "save" and decision.category:
        saved = save_pair(
            analyses,
            category=decision.category,
            output_root=output_root,
            class_ids=class_ids,
            live_timestamp=live_timestamp,
        )
        if decision.category == "accepted" and decision.number:
            update_state(
                state_path,
                {
                    "last_accepted_number": decision.number,
                },
            )
        elif decision.category and decision.category.startswith("manual/"):
            append_recognition_info(
                output_root=output_root,
                category=decision.category,
                live_timestamp=live_timestamp,
                analyses=analyses,
                saved=saved,
            )
            update_state(
                state_path,
                {
                    "last_manual_key": manual_duplicate_key(analyses),
                },
            )

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "decision": decision.action,
        "reason": decision.reason,
        "category": decision.category,
        "number": decision.number,
        "saved": saved,
        "cameras": [analysis_record(analysis) for analysis in analyses],
    }
    append_log(log_path, record)
    print_iteration(decision, analyses)
    return decision


def process_single_camera_iteration(
    captures: Sequence[Capture],
    *,
    model: YOLO,
    class_ids: dict[str, int],
    output_root: Path,
    state_path: Path,
    log_path: Path,
    min_detections: int,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: float | None,
    mode: str,
    live_timestamp: datetime | None,
) -> Decision:
    if len(captures) != 1:
        raise ValueError("single_camera mode expects exactly one capture")

    state = load_state(state_path)
    analyses = analyse_captures(
        captures,
        model=model,
        conf=conf,
        iou=iou,
        max_det=max_det,
        merge_iou=merge_iou,
    )
    analysis = analyses[0]
    decision = decide_single_camera(
        analysis,
        min_detections=min_detections,
        last_number=normalize_number(state.get("last_accepted_number")),
        last_manual_key=str(state.get("last_manual_key") or ""),
    )

    saved: list[dict[str, str]] = []
    if decision.action == "save" and decision.category:
        saved = save_pair(
            analyses,
            category=decision.category,
            output_root=output_root,
            class_ids=class_ids,
            live_timestamp=live_timestamp,
        )
        if decision.category == "accepted" and decision.number:
            update_state(state_path, {"last_accepted_number": decision.number})
        elif decision.category and decision.category.startswith("manual/"):
            append_recognition_info(
                output_root=output_root,
                category=decision.category,
                live_timestamp=live_timestamp,
                analyses=analyses,
                saved=saved,
            )
            update_state(state_path, {"last_manual_key": manual_duplicate_key(analyses)})

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "decision": decision.action,
        "reason": decision.reason,
        "category": decision.category,
        "number": decision.number,
        "saved": saved,
        "cameras": [analysis_record(item) for item in analyses],
    }
    append_log(log_path, record)
    print_iteration(decision, analyses)
    return decision


def process_single_camera_fallback(
    captures: Sequence[Capture],
    *,
    model: YOLO,
    class_ids: dict[str, int],
    output_root: Path,
    state_path: Path,
    log_path: Path,
    min_detections: int,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: float | None,
    live_timestamp: datetime,
    camera_errors: Sequence[dict[str, str]],
) -> Decision:
    analyses = analyse_captures(
        captures,
        model=model,
        conf=conf,
        iou=iou,
        max_det=max_det,
        merge_iou=merge_iou,
    )

    state = load_state(state_path)
    manual_key = manual_duplicate_key(analyses)
    has_container = any(analysis.detections_count >= min_detections for analysis in analyses)
    if has_container and manual_key == str(state.get("last_single_camera_manual_key") or ""):
        decision = Decision("skip", "duplicate_single_camera_manual_case", None, None)
        saved = []
    elif has_container:
        decision = Decision("save", "single_camera_other_camera_error", None, "single_camera/manual")
        saved = save_pair(
            analyses,
            category=decision.category,
            output_root=output_root,
            class_ids=class_ids,
            live_timestamp=live_timestamp,
        )
        update_state(
            state_path,
            {
                "last_single_camera_manual_key": manual_key,
            },
        )
    else:
        decision = Decision("skip", "single_camera_not_enough_detections", None, None)
        saved = []

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "live_single_camera_fallback",
        "decision": decision.action,
        "reason": decision.reason,
        "category": decision.category,
        "number": decision.number,
        "saved": saved,
        "camera_errors": list(camera_errors),
        "cameras": [analysis_record(analysis) for analysis in analyses],
    }
    append_log(log_path, record)
    print_iteration(decision, analyses)
    for error in camera_errors:
        print(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"CAMERA_ERROR {error['camera']} {error['error']}",
            flush=True,
        )
    return decision


def main() -> int:
    global ACTIVE_RECOGNIZER_MODULE

    args = parse_args()
    config = load_json(args.config.resolve())
    recognizer_module_name = args.recognizer_module or config.get("recognizer_module") or "read_container"
    ACTIVE_RECOGNIZER_MODULE = load_recognizer_module(recognizer_module_name)
    collector_mode = parse_collector_mode(config)
    cameras = parse_camera_configs(config, collector_mode)

    source_args = [args.source1, args.source2]
    if collector_mode == COLLECTOR_MODE_SINGLE:
        if any(source_args):
            raise SystemExit("Use --source in single_camera mode, not --source1/--source2")
    else:
        if args.source is not None:
            raise SystemExit("Use --source1 and --source2 in dual_camera mode, not --source")
        if any(source_args) and not all(source_args):
            raise SystemExit("--source1 and --source2 must be passed together")

    weights = resolve_repo_path(config.get("weights"), DEFAULT_WEIGHTS)
    output_root = resolve_repo_path(args.output_root or config.get("output_root"), DEFAULT_OUTPUT_ROOT)
    state_path = resolve_repo_path(args.state_file or config.get("state_file"), DEFAULT_STATE_FILE)
    log_path = resolve_repo_path(args.log_jsonl or config.get("log_jsonl"), DEFAULT_LOG_JSONL)
    min_detections = int(args.min_detections or config.get("min_detections", 6))
    interval_minutes = float(args.interval_minutes or config.get("interval_minutes", 20.0))
    require_both_detections = bool(config.get("require_both_detections", False))
    if args.require_both_detections:
        require_both_detections = True
    conf = float(config.get("conf", 0.15))
    iou = float(config.get("iou", 0.45))
    max_det = int(config.get("max_det", 300))
    merge_iou_raw = float(config.get("merge_iou", recognizer_default_merge_iou(ACTIVE_RECOGNIZER_MODULE)))
    merge_iou = None if merge_iou_raw <= 0 else merge_iou_raw
    save_single_camera_on_error = bool(config.get("save_single_camera_on_error", True))
    if args.no_single_camera_on_error:
        save_single_camera_on_error = False

    model = YOLO(str(weights))
    class_ids = build_class_id_map(model)

    if collector_mode == COLLECTOR_MODE_SINGLE and args.source is not None:
        captures = local_single_capture(cameras[0], args.source)
        process_single_camera_iteration(
            captures,
            model=model,
            class_ids=class_ids,
            output_root=output_root,
            state_path=state_path,
            log_path=log_path,
            min_detections=min_detections,
            conf=conf,
            iou=iou,
            max_det=max_det,
            merge_iou=merge_iou,
            mode="local_single_camera",
            live_timestamp=None,
        )
        return 0

    if collector_mode == COLLECTOR_MODE_DUAL and args.source1 is not None and args.source2 is not None:
        captures = local_captures(cameras, args.source1, args.source2)
        process_iteration(
            captures,
            model=model,
            class_ids=class_ids,
            output_root=output_root,
            state_path=state_path,
            log_path=log_path,
            min_detections=min_detections,
            conf=conf,
            iou=iou,
            max_det=max_det,
            merge_iou=merge_iou,
            require_both_detections=require_both_detections,
            mode="local",
            live_timestamp=None,
        )
        return 0

    interval_seconds = max(1.0, interval_minutes * 60.0)
    print(
        f"Live mode: {collector_mode}; recognizer={ACTIVE_RECOGNIZER_MODULE.__name__}; "
        f"polling {len(cameras)} camera(s) every {interval_minutes:g} minutes; "
        f"output={output_root}",
        flush=True,
    )

    while True:
        live_timestamp = datetime.now()
        try:
            with tempfile.TemporaryDirectory(prefix="axis_dual_container_") as tmp:
                tmp_path = Path(tmp)
                if collector_mode == COLLECTOR_MODE_SINGLE:
                    captures = live_captures(cameras, tmp_path)
                    process_single_camera_iteration(
                        captures,
                        model=model,
                        class_ids=class_ids,
                        output_root=output_root,
                        state_path=state_path,
                        log_path=log_path,
                        min_detections=min_detections,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        merge_iou=merge_iou,
                        mode="live_single_camera",
                        live_timestamp=live_timestamp,
                    )
                elif save_single_camera_on_error:
                    captures, camera_errors = live_captures_partial(cameras, tmp_path)
                    if len(captures) == len(cameras):
                        process_iteration(
                            captures,
                            model=model,
                            class_ids=class_ids,
                            output_root=output_root,
                            state_path=state_path,
                            log_path=log_path,
                            min_detections=min_detections,
                            conf=conf,
                            iou=iou,
                            max_det=max_det,
                            merge_iou=merge_iou,
                            require_both_detections=require_both_detections,
                            mode="live",
                            live_timestamp=live_timestamp,
                        )
                    elif captures:
                        process_single_camera_fallback(
                            captures,
                            model=model,
                            class_ids=class_ids,
                            output_root=output_root,
                            state_path=state_path,
                            log_path=log_path,
                            min_detections=min_detections,
                            conf=conf,
                            iou=iou,
                            max_det=max_det,
                            merge_iou=merge_iou,
                            live_timestamp=live_timestamp,
                            camera_errors=camera_errors,
                        )
                    else:
                        raise RuntimeError(
                            "; ".join(
                                f"{error['camera']}: {error['error']}" for error in camera_errors
                            )
                            or "all cameras failed"
                        )
                else:
                    captures = live_captures(cameras, tmp_path)
                    process_iteration(
                        captures,
                        model=model,
                        class_ids=class_ids,
                        output_root=output_root,
                        state_path=state_path,
                        log_path=log_path,
                        min_detections=min_detections,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        merge_iou=merge_iou,
                        require_both_detections=require_both_detections,
                        mode="live",
                        live_timestamp=live_timestamp,
                    )
        except KeyboardInterrupt:
            print("Stopped by user.", flush=True)
            return 0
        except Exception as exc:
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "mode": "live",
                "decision": "error",
                "error": str(exc),
            }
            append_log(log_path, record)
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ERROR {exc}", flush=True)

        if args.once:
            return 0
        time.sleep(interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
