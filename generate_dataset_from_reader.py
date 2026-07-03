#!/usr/bin/env python3
"""
Интерактивный скрипт для генерации датасета в формате YOK1_dataset путём запуска read_container.py.

Отображает изображение с результатом распознавания и позволяет подтвердить (Y) или отклонить (N).

Формат выходного dataset.jsonl:
{
    "id": "camera1_01-17-02",
    "image": "images/camera1_01-17-02.jpg",
    "scenario": "extract_number_general",
    "expected": {
        "owner_code": "TGHU",
        "registration_number": "095205",
        "check_digit": "0",
        "type_size_code": ""
    },
    "metadata": {
        "scenario_version": 1,
        "iso_code": "TGHU0952050",
        "full_code": "TGHU0952050",
        "source_image": "C:\\path\\to\\source.jpg",
        "model": "C:\\path\\to\\model.pt",
        "latency_ms": 472.778,
        "raw_detections_count": 11,
        "detections_count": 11,
        "duplicates_removed": 0
    }
}

Управление:
- Y / y / Enter / Space — сохранить (распознано правильно)
- N / n — пропустить (распознано неправильно)
- Q / q / Esc — выйти
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import cv2
    HAS_CV = True
except ImportError:
    HAS_CV = False

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TYPE_SIZE_RE = re.compile(r"^[0-9]{2}[A-Z0-9][0-9]$")
PREVIEW_WINDOW_NAME = "Dataset Builder"
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_PAGE_UP = 2162688
KEY_PAGE_DOWN = 2228224
ISO6346_VALUES = {
    **{str(i): i for i in range(10)},
    "A": 10,
    "B": 12,
    "C": 13,
    "D": 14,
    "E": 15,
    "F": 16,
    "G": 17,
    "H": 18,
    "I": 19,
    "J": 20,
    "K": 21,
    "L": 23,
    "M": 24,
    "N": 25,
    "O": 26,
    "P": 27,
    "Q": 28,
    "R": 29,
    "S": 30,
    "T": 31,
    "U": 32,
    "V": 34,
    "W": 35,
    "X": 36,
    "Y": 37,
    "Z": 38,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Интерактивная генерация датасета в формате YOK1_dataset через запуск read_container.py"
    )
    parser.add_argument(
        "--reader",
        type=Path,
        help="Путь к read_container.py (нужен только без --collector-jsonl)"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Путь к YOLO модели (.pt) (нужен только без --collector-jsonl)"
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Папка с изображениями или glob-паттерн (нужен только без --collector-jsonl)"
    )
    parser.add_argument(
        "--collector-jsonl",
        nargs="+",
        type=Path,
        default=None,
        help="JSONL логи axis_dual_container_collector.py; используются уже сохраненные детекции"
    )
    parser.add_argument(
        "--collector-pairs",
        action="store_true",
        help="В режиме --collector-jsonl создавать один evaluate_accuracy case на пару cam1+cam2"
    )
    parser.add_argument(
        "--valid-type-codes",
        type=Path,
        default=Path("configs/valid_type_size_codes.txt"),
        help="Allowlist type_size_code, один код на строку"
    )
    parser.add_argument(
        "--include-unsaved-collector-images",
        action="store_true",
        help="В режиме --collector-jsonl брать cam.image, если запись не была сохранена коллектором"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Выходная папка для датасета"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Рекурсивный поиск изображений"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Порог уверенности YOLO (по умолчанию 0.15 как в read_container.py)"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Порог NMS IoU"
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Максимум детекций на изображение"
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.35,
        help="Порог слияния дублей (0 для отключения)"
    )
    parser.add_argument(
        "--scenario-name",
        default="extract_number_general",
        help="Имя сценария"
    )
    parser.add_argument(
        "--scenario-version",
        type=int,
        default=1,
        help="Версия сценария"
    )
    parser.add_argument(
        "--dataset-file",
        default="dataset.jsonl",
        help="Имя файла датасета"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Копировать изображения в output_dir/images/"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Ограничение количества изображений"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Неинтерактивный режим: сохранять все результаты автоматически"
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.0,
        help="Масштаб окна предпросмотра (по умолчанию 1.0)"
    )
    return parser.parse_args()


def collect_images(source: Path, recursive: bool = False) -> List[Path]:
    """Собрать список изображений из папки."""
    if source.is_file():
        return [source.resolve()] if source.suffix.lower() in IMAGE_SUFFIXES else []
    
    if source.is_dir():
        if recursive:
            return sorted(
                p.resolve() for p in source.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        return sorted(
            p.resolve() for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    
    # Glob паттерн
    if "*" in str(source):
        return sorted(p.resolve() for p in Path(".").glob(str(source)) if p.suffix.lower() in IMAGE_SUFFIXES)
    
    return []


def normalize_text(value: Any) -> str:
    """Normalize OCR text for container-number/type fields."""
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def split_expected(number: str, type_size_code: str = "") -> Dict[str, str]:
    """Build dataset expected fields from an ISO 6346 number and type/size code."""
    code = normalize_text(number)
    return {
        "owner_code": code[:4] if len(code) >= 4 else code,
        "registration_number": code[4:10] if len(code) >= 10 else code[4:],
        "check_digit": code[10:11] if len(code) >= 11 else "",
        "type_size_code": normalize_text(type_size_code),
    }


def load_valid_type_codes(path: Path) -> Set[str]:
    """Load valid type/size codes from a text allowlist."""
    if not path.exists():
        return set()
    codes: Set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        code = normalize_text(line.split("#", 1)[0])
        if code:
            codes.add(code)
    return codes


def type_size_check(type_size_code: str, valid_codes: Set[str]) -> Dict[str, bool]:
    code = normalize_text(type_size_code)
    format_ok = bool(TYPE_SIZE_RE.fullmatch(code)) if code else False
    allowlist_ok = code in valid_codes if code else False
    return {
        "present": bool(code),
        "format_ok": format_ok,
        "allowlist_ok": allowlist_ok,
        "ok": allowlist_ok if valid_codes else format_ok,
    }


def iso6346_check_ok(number: str) -> Optional[bool]:
    code = normalize_text(number)
    if len(code) != 11 or not code[:4].isalpha() or not code[4:].isdigit():
        return False
    total = 0
    for idx, char in enumerate(code[:10]):
        value = ISO6346_VALUES.get(char)
        if value is None:
            return False
        total += value * (2 ** idx)
    expected_digit = (total % 11) % 10
    return expected_digit == int(code[10])


def read_jsonl_records(paths: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    print(f"[warn] bad JSON in {path}:{line_no}: {exc}")
                    continue
                if isinstance(row, dict):
                    row["_source_jsonl"] = str(path)
                    row["_source_line"] = line_no
                    records.append(row)
    return records


def collector_samples(
    jsonl_paths: List[Path],
    include_unsaved_images: bool,
    valid_type_codes: Set[str],
) -> List[Dict[str, Any]]:
    """Convert collector JSONL camera rows into dataset-builder samples."""
    samples: List[Dict[str, Any]] = []
    records = read_jsonl_records(jsonl_paths)
    for record in records:
        saved_by_camera = {
            str(item.get("camera") or ""): str(item.get("image") or "")
            for item in record.get("saved", [])
            if isinstance(item, dict)
        }
        cameras = record.get("cameras", [])
        if not isinstance(cameras, list):
            continue
        for cam in cameras:
            if not isinstance(cam, dict):
                continue
            camera_name = str(cam.get("camera") or "camera")
            saved_image_text = saved_by_camera.get(camera_name) or ""
            image_text = saved_image_text or (
                str(cam.get("image") or "") if include_unsaved_images else ""
            )
            if not image_text:
                continue
            image_path = Path(image_text)
            if not image_path.exists() and saved_image_text and include_unsaved_images:
                image_text = str(cam.get("image") or "")
                image_path = Path(image_text)
            if not image_path.exists():
                print(f"[warn] collector image not found, skip: {image_path}")
                continue

            number = normalize_text(cam.get("primary_number"))
            type_size = normalize_text(cam.get("size_type_code"))
            type_status = type_size_check(type_size, valid_type_codes)
            expected = split_expected(number, type_size)
            metadata = {
                "source_image": str(image_path),
                "collector_jsonl": record.get("_source_jsonl", ""),
                "collector_line": record.get("_source_line", ""),
                "collector_timestamp": record.get("timestamp", ""),
                "collector_category": record.get("category", ""),
                "collector_reason": record.get("reason", ""),
                "camera": camera_name,
                "check_ok": bool(cam.get("check_ok")),
                "type_size_present": type_status["present"],
                "type_size_format_ok": type_status["format_ok"],
                "type_size_allowlist_ok": type_status["allowlist_ok"],
                "type_size_ok": type_status["ok"],
                "raw_detections_count": int(cam.get("detections_count") or 0),
                "detections_count": int(cam.get("detections_count") or 0),
                "duplicates_removed": 0,
                "layout": cam.get("layout", ""),
            }
            result = {
                "result": format_container_result_like_reader(expected),
                "elapsed_ms": 0.0,
                "raw_detections_count": metadata["raw_detections_count"],
                "detections_count": metadata["detections_count"],
                "duplicates_removed": 0,
                "check_ok": metadata["check_ok"],
                "type_size_ok": metadata["type_size_ok"],
                "type_size_format_ok": metadata["type_size_format_ok"],
                "type_size_allowlist_ok": metadata["type_size_allowlist_ok"],
            }
            samples.append(
                {
                    "image_path": image_path.resolve(),
                    "case_stem": f"{image_path.stem}_{camera_name}",
                    "expected": expected,
                    "result": result,
                    "metadata": metadata,
                }
            )
    return samples


def format_container_result_like_reader(expected: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {"label": "owner_code", "text": expected.get("owner_code", "")},
        {"label": "registration_number", "text": expected.get("registration_number", "")},
        {"label": "check_digit", "text": expected.get("check_digit", "")},
        {"label": "type_size_code", "text": expected.get("type_size_code", "")},
    ]


def load_existing_iso_codes(dataset_path: Path) -> Set[str]:
    codes: Set[str] = set()
    if not dataset_path.exists():
        return codes
    for line in dataset_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        expected = row.get("expected")
        if isinstance(expected, dict):
            code = normalize_text(
                f"{expected.get('owner_code', '')}"
                f"{expected.get('registration_number', '')}"
                f"{expected.get('check_digit', '')}"
            )
            if code:
                codes.add(code)
    return codes


def collector_image_for_camera(
    record: Dict[str, Any],
    cam: Dict[str, Any],
    include_unsaved_images: bool,
) -> Optional[Path]:
    camera_name = str(cam.get("camera") or "")
    saved_items = record.get("saved") if isinstance(record.get("saved"), list) else []
    saved_by_camera = {
        str(item.get("camera") or ""): str(item.get("image") or "")
        for item in saved_items
        if isinstance(item, dict)
    }
    candidates: List[str] = []
    if saved_by_camera.get(camera_name):
        candidates.append(saved_by_camera[camera_name])
    if include_unsaved_images and cam.get("image"):
        candidates.append(str(cam.get("image")))
    for value in candidates:
        path = Path(value)
        if path.exists():
            return path.resolve()
    return None


def choose_pair_number(record: Dict[str, Any], cameras: List[Dict[str, Any]]) -> str:
    record_number = normalize_text(record.get("number"))
    if record_number:
        return record_number
    check_ok_numbers = [
        normalize_text(cam.get("primary_number"))
        for cam in cameras
        if bool(cam.get("check_ok")) and normalize_text(cam.get("primary_number"))
    ]
    unique = sorted(set(check_ok_numbers))
    if len(unique) == 1:
        return unique[0]
    return ""


def choose_pair_type_size(
    cameras: List[Dict[str, Any]],
    valid_type_codes: Set[str],
) -> tuple[str, str]:
    candidates: List[tuple[int, str, str]] = []
    for cam in cameras:
        code = normalize_text(cam.get("size_type_code"))
        if not code:
            continue
        status = type_size_check(code, valid_type_codes)
        score = 0
        if status["allowlist_ok"]:
            score += 100
        if status["format_ok"]:
            score += 10
        if bool(cam.get("check_ok")):
            score += 1
        candidates.append((score, code, str(cam.get("camera") or "")))
    if not candidates:
        return "", ""
    candidates.sort(reverse=True)
    _score, code, camera_name = candidates[0]
    return code, camera_name


def collector_pair_samples(
    jsonl_paths: List[Path],
    include_unsaved_images: bool,
    valid_type_codes: Set[str],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    records = read_jsonl_records(jsonl_paths)
    for record in records:
        cameras = [cam for cam in record.get("cameras", []) if isinstance(cam, dict)]
        if len(cameras) < 2:
            continue
        pair_cameras: List[Dict[str, Any]] = []
        image_paths: List[Path] = []
        for cam in cameras[:2]:
            image_path = collector_image_for_camera(record, cam, include_unsaved_images)
            if image_path is None:
                break
            pair_cameras.append(cam)
            image_paths.append(image_path)
        if len(image_paths) != 2:
            continue

        pair_number = choose_pair_number(record, pair_cameras)
        pair_type, type_source = choose_pair_type_size(pair_cameras, valid_type_codes)
        expected = split_expected(pair_number, pair_type)
        samples.append(
            {
                "case_stem": (
                    f"pair_{record.get('_source_line', len(samples) + 1)}_"
                    f"{image_paths[0].stem}_{image_paths[1].stem}"
                ),
                "image_paths": image_paths,
                "expected": expected,
                "record": record,
                "cameras": pair_cameras,
                "type_source_camera": type_source,
            }
        )
    return samples


def make_pair_preview(sample: Dict[str, Any], expected: Dict[str, str]) -> Optional[Any]:
    if not HAS_CV:
        return None
    frames = []
    for path in sample["image_paths"]:
        frame = cv2.imread(str(path))
        if frame is None:
            return None
        frames.append(frame)
    target_h = min(720, max(frame.shape[0] for frame in frames))
    resized = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = target_h / max(h, 1)
        resized.append(cv2.resize(frame, (max(1, int(w * scale)), target_h)))
    canvas = cv2.hconcat(resized)

    iso_code = build_iso_code(expected)
    type_size = expected.get("type_size_code", "")
    type_status = type_size_check(type_size, set())
    record = sample["record"]
    cameras = sample["cameras"]
    lines = [
        f"PAIR ISO: {iso_code or '-'}  type_size: {type_size or '-'}",
        f"pair_check_ok: {iso6346_check_ok(iso_code)}  type_format_ok: {type_status['format_ok']}",
        f"decision: {record.get('decision', '-')}  reason: {record.get('reason', '-')}",
    ]
    for cam in cameras:
        lines.append(
            f"{cam.get('camera', '-')}: num={normalize_text(cam.get('primary_number')) or '-'} "
            f"check_ok={bool(cam.get('check_ok'))} "
            f"type={normalize_text(cam.get('size_type_code')) or '-'} "
            f"dets={cam.get('detections_count', 0)}"
        )
    lines.extend(
        [
            "Y/Enter/Space accept | E edit | N reject",
            "Left/A previous | Right/D next | Q/Esc quit",
        ]
    )

    font_scale = min(0.9, max(0.5, canvas.shape[1] / 1500))
    thickness = max(1, int(2 * font_scale))
    for i, line in enumerate(lines):
        color = (100, 255, 100) if i == 0 else (220, 240, 255)
        origin = (15, 25 + i * 28)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return canvas


def show_pair_preview(sample: Dict[str, Any], expected: Dict[str, str]) -> Optional[str]:
    frame = make_pair_preview(sample, expected)
    if frame is None:
        return None
    cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(PREVIEW_WINDOW_NAME, frame)
    key = cv2.waitKeyEx(0)
    key_char = key & 0xFF
    if key_char in (ord("q"), ord("Q"), 27):
        return "quit"
    if key_char in (ord("y"), ord("Y"), 13, 32):
        return "save"
    if key_char in (ord("n"), ord("N")):
        return "skip"
    if key_char in (ord("e"), ord("E")):
        return "edit"
    if key_char in (ord("a"), ord("A"), ord("b"), ord("B"), ord("p"), ord("P")) or key in (KEY_LEFT, KEY_PAGE_UP):
        return "prev"
    if key_char in (ord("d"), ord("D")) or key in (KEY_RIGHT, KEY_PAGE_DOWN):
        return "next"
    return "refresh"


def save_pair_case(
    *,
    image_paths: List[Path],
    output_images_dir: Path,
    dataset_path: Path,
    case_id: str,
    scenario_name: str,
    scenario_version: int,
    expected: Dict[str, str],
    metadata: Dict[str, Any],
    copy_images: bool,
) -> None:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    image_refs: List[str] = []
    for idx, image_path in enumerate(image_paths, start=1):
        if copy_images:
            suffix = image_path.suffix.lower() or ".jpg"
            out_name = f"{case_id}_cam{idx}{suffix}"
            out_path = output_images_dir / out_name
            shutil.copy2(image_path, out_path)
            image_refs.append(f"images/{out_name}")
        else:
            image_refs.append(str(image_path.resolve()))
    iso_code = build_iso_code(expected)
    row = {
        "id": case_id,
        "images": image_refs,
        "scenario": scenario_name,
        "expected": expected,
        "metadata": {
            "scenario_version": scenario_version,
            "iso_code": iso_code,
            "full_code": build_full_code(expected),
            **metadata,
        },
    }
    with dataset_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_collector_pair_builder(
    args: argparse.Namespace,
    *,
    output_images_dir: Path,
    dataset_path: Path,
    valid_type_codes: Set[str],
) -> Dict[str, Any]:
    samples = collector_pair_samples(
        [path.resolve() for path in args.collector_jsonl],
        include_unsaved_images=args.include_unsaved_collector_images,
        valid_type_codes=valid_type_codes,
    )
    if args.max_images is not None:
        samples = samples[: args.max_images]
    if not samples:
        raise RuntimeError("No complete cam1+cam2 pairs found in collector JSONL.")

    existing_ids = load_existing_case_ids(dataset_path)
    accepted_numbers = load_existing_iso_codes(dataset_path)
    processed = saved = skipped = duplicates = errors = 0
    idx = 0
    while idx < len(samples):
        sample = samples[idx]
        expected = dict(sample["expected"])
        print(
            f"\n[{idx + 1}/{len(samples)}] pair "
            f"{sample['image_paths'][0].name} + {sample['image_paths'][1].name}"
        )
        print(
            f"[pair] ISO={build_iso_code(expected) or '-'} "
            f"type={expected.get('type_size_code') or '-'}"
        )

        if args.no_interactive:
            decision = "save"
        else:
            decision = show_pair_preview(sample, expected)

        if decision == "quit":
            break
        if decision == "prev":
            idx = max(0, idx - 1)
            continue
        if decision == "next":
            idx = min(len(samples) - 1, idx + 1)
            continue
        if decision == "refresh":
            continue
        if decision == "edit":
            expected = edit_expected_in_console(expected, valid_type_codes)
            sample["expected"] = expected
            decision = "save"
        if decision == "skip":
            skipped += 1
            idx += 1
            continue
        if decision != "save":
            idx += 1
            continue

        processed += 1
        final_number = build_iso_code(expected)
        if not final_number:
            print("[skip] empty final ISO number")
            skipped += 1
            idx += 1
            continue
        if final_number in accepted_numbers:
            print(f"[duplicate] {final_number} already exists; pair skipped")
            duplicates += 1
            idx += 1
            continue

        case_id = ensure_unique_case_id(str(sample.get("case_stem") or final_number), existing_ids)
        existing_ids.add(case_id)
        type_status = type_size_check(expected.get("type_size_code", ""), valid_type_codes)
        metadata = {
            "collector_jsonl": sample["record"].get("_source_jsonl", ""),
            "collector_line": sample["record"].get("_source_line", ""),
            "collector_timestamp": sample["record"].get("timestamp", ""),
            "collector_decision": sample["record"].get("decision", ""),
            "collector_reason": sample["record"].get("reason", ""),
            "camera_numbers": {
                str(cam.get("camera") or f"cam{i + 1}"): normalize_text(cam.get("primary_number"))
                for i, cam in enumerate(sample["cameras"])
            },
            "camera_check_ok": {
                str(cam.get("camera") or f"cam{i + 1}"): bool(cam.get("check_ok"))
                for i, cam in enumerate(sample["cameras"])
            },
            "camera_type_size_codes": {
                str(cam.get("camera") or f"cam{i + 1}"): normalize_text(cam.get("size_type_code"))
                for i, cam in enumerate(sample["cameras"])
            },
            "type_size_source_camera": sample.get("type_source_camera", ""),
            "final_check_ok": iso6346_check_ok(final_number),
            "final_type_size_present": type_status["present"],
            "final_type_size_format_ok": type_status["format_ok"],
            "final_type_size_allowlist_ok": type_status["allowlist_ok"],
            "final_type_size_ok": type_status["ok"],
        }
        try:
            save_pair_case(
                image_paths=sample["image_paths"],
                output_images_dir=output_images_dir,
                dataset_path=dataset_path,
                case_id=case_id,
                scenario_name=args.scenario_name,
                scenario_version=args.scenario_version,
                expected=expected,
                metadata=metadata,
                copy_images=args.copy_images,
            )
        except Exception as exc:
            errors += 1
            print(f"[error] failed to save pair: {exc}")
            idx += 1
            continue
        accepted_numbers.add(final_number)
        saved += 1
        print(f"[save] {case_id} | ISO={final_number} type={expected.get('type_size_code', '')}")
        idx += 1

    if not args.no_interactive and HAS_CV:
        cv2.destroyAllWindows()
    return {
        "mode": "collector_pairs",
        "pairs_total": len(samples),
        "processed": processed,
        "saved": saved,
        "skipped": skipped,
        "duplicates": duplicates,
        "errors": errors,
    }


def run_reader(
    reader_path: Path,
    model_path: Path,
    image_path: Path,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: Optional[float]
) -> Optional[Dict[str, Any]]:
    """Запустить read_container.py для одного изображения и вернуть результат."""
    cmd = [
        sys.executable,
        str(reader_path),
        "--weights", str(model_path),
        "--source", str(image_path),
        "--conf", str(conf),
        "--iou", str(iou),
        "--max-det", str(max_det),
    ]
    
    if merge_iou is not None and merge_iou > 0:
        cmd.extend(["--merge-iou", str(merge_iou)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(reader_path.parent)
        )
        
        if result.returncode != 0:
            print(f"[error] read_container.py вернул код {result.returncode}: {result.stderr.strip()}")
            return None
        
        output = result.stdout.strip()
        if not output:
            return None
        
        # read_container.py выводит JSON строку
        data = json.loads(output)
        return data
        
    except subprocess.TimeoutExpired:
        print(f"[warn] таймаут для {image_path.name}")
        return None
    except json.JSONDecodeError as e:
        print(f"[warn] не удалось распарсить JSON вывод: {e}")
        return None
    except Exception as e:
        print(f"[error] ошибка при запуске: {e}")
        return None


def parse_reader_result(data: Dict[str, Any]) -> Dict[str, str]:
    """Преобразовать результат read_container.py в формат expected полей."""
    result_list = data.get("result", [])
    
    fields = {
        "owner_code": "",
        "registration_number": "",
        "check_digit": "",
        "type_size_code": ""
    }
    
    for item in result_list:
        if isinstance(item, dict):
            label = item.get("label", "")
            text = item.get("text", "")
            if label in fields:
                fields[label] = text
    
    return fields


def build_iso_code(expected: Dict[str, str]) -> str:
    """Построить ISO код из полей."""
    return (
        f"{expected.get('owner_code', '')}"
        f"{expected.get('registration_number', '')}"
        f"{expected.get('check_digit', '')}"
    )


def build_full_code(expected: Dict[str, str]) -> str:
    """Построить полный код с type_size_code."""
    iso = build_iso_code(expected)
    type_size = expected.get("type_size_code", "")
    if type_size:
        return f"{iso} {type_size}".strip()
    return iso.strip()


def edit_expected_in_console(expected: Dict[str, str], valid_type_codes: Set[str]) -> Dict[str, str]:
    """Let the user correct number/type in the terminal while the preview is paused."""
    current_number = build_iso_code(expected)
    current_type = expected.get("type_size_code", "")
    print("\n[edit] Press Enter to keep the current value, '-' to clear it.")
    number = input(f"[edit] container number [{current_number}]: ").strip()
    if number == "":
        number = current_number
    elif number == "-":
        number = ""
    type_size = input(f"[edit] type_size [{current_type}]: ").strip()
    if type_size == "":
        type_size = current_type
    elif type_size == "-":
        type_size = ""

    corrected = split_expected(number, type_size)
    status = type_size_check(corrected.get("type_size_code", ""), valid_type_codes)
    print(
        "[edit] corrected: "
        f"ISO={build_iso_code(corrected) or '(empty)'}, "
        f"type_size={corrected.get('type_size_code', '') or '(empty)'}, "
        f"type_ok={status['ok']}"
    )
    return corrected


def save_case(
    source_image: Path,
    output_images_dir: Path,
    dataset_path: Path,
    case_id: str,
    scenario_name: str,
    scenario_version: int,
    expected: Dict[str, str],
    metadata: Dict[str, Any],
    copy_images: bool = False
) -> None:
    """Сохранить кейс в датасет."""
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    if copy_images:
        suffix = source_image.suffix.lower() or ".jpg"
        out_image_name = f"{case_id}{suffix}"
        out_image_path = output_images_dir / out_image_name
        shutil.copy2(source_image, out_image_path)
        image_ref = f"images/{out_image_name}"
    else:
        # Используем абсолютный путь к исходному изображению
        image_ref = str(source_image.resolve())
    
    iso_code = build_iso_code(expected)
    full_code = build_full_code(expected)
    
    row = {
        "id": case_id,
        "image": image_ref,
        "scenario": scenario_name,
        "expected": expected,
        "metadata": {
            "scenario_version": scenario_version,
            "iso_code": iso_code,
            "full_code": full_code,
            **metadata,
        },
    }
    
    with dataset_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_unique_case_id(stem: str, existing: Set[str]) -> str:
    """Убедиться, что ID уникален."""
    normalized = stem.strip() or "sample"
    if normalized not in existing:
        return normalized
    
    idx = 2
    while True:
        candidate = f"{normalized}_{idx}"
        if candidate not in existing:
            return candidate
        idx += 1


def load_existing_case_ids(dataset_path: Path) -> Set[str]:
    """Загрузить существующие ID кейсов."""
    if not dataset_path.exists():
        return set()
    
    ids: Set[str] = set()
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = row.get("id")
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    
    return ids


def create_annotated_image(
    image_path: Path,
    result: Dict[str, Any],
    expected: Dict[str, str]
) -> Optional[Any]:
    """Создать изображение с аннотацией результата."""
    if not HAS_CV:
        return None
    
    # Загрузка изображения
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    
    # Получение размеров для масштабирования текста
    _height, width = frame.shape[:2]
    font_scale = min(1.0, max(0.5, width / 800))
    thickness = max(1, int(2 * font_scale))
    
    # Извлечение результата распознавания
    result_list = result.get("result", [])
    recognized_text = ""
    for item in result_list:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                recognized_text += text
    
    elapsed_ms = result.get("elapsed_ms", 0)
    
    # Формирование строк для отображения
    iso_code = build_iso_code(expected)
    full_code = build_full_code(expected)
    type_size = expected.get("type_size_code", "")
    check_ok = result.get("check_ok")
    type_size_ok = result.get("type_size_ok")
    check_text = "-" if check_ok is None else str(bool(check_ok))
    type_check_text = "-" if type_size_ok is None else str(bool(type_size_ok))
    
    lines = [
        f"ISO: {iso_code}",
        f"Full: {full_code}",
        f"Type/Size: {type_size}",
        f"check_ok: {check_text} | type_size_ok: {type_check_text}",
        f"Time: {elapsed_ms:.1f}ms",
        "",
        "Y/Enter/Space = сохранить",
        "E = исправить номер/type_size и сохранить",
        "N = пропустить (неправильно)",
        "Q/Esc = выход"
    ]
    
    # Рисование фона для текста
    # Рисование текста
    y_offset = 25
    for i, line in enumerate(lines):
        color = (100, 255, 100) if i == 0 else (200, 255, 200) if i < 3 else (180, 180, 255)
        if i >= 4:  # Инструкции
            color = (200, 200, 255)
        origin = (15, y_offset + i * 30)
        cv2.putText(
            frame,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.7,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            line,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.7,
            color,
            thickness,
            cv2.LINE_AA
        )
    
    return frame


def show_preview(image_path: Path, result: Dict[str, Any], expected: Dict[str, str]) -> Optional[str]:
    """
    Показать предпросмотр с результатом распознавания.
    Возвращает "save", "skip", "edit" или None если выйти.
    """
    if not HAS_CV:
        print("[warn] OpenCV не установлен. Используйте --no-interactive для автоматического режима.")
        return None
    
    frame = create_annotated_image(image_path, result, expected)
    if frame is None:
        return None
    
    window_name = "Dataset Builder - Y/N для подтверждения"
    cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(PREVIEW_WINDOW_NAME, frame)
    
    # Ожидание нажатия клавиши
    key = cv2.waitKeyEx(0)
    key_char = key & 0xFF
    
    if key_char in (ord("q"), ord("Q"), 27):  # Q, q, Esc
        return "quit"
    elif key_char in (ord("y"), ord("Y")):
        return "save"
    elif key_char in (ord("n"), ord("N")):
        return "skip"
    elif key_char in (ord("e"), ord("E")):
        return "edit"
    elif key_char in (ord("a"), ord("A"), ord("b"), ord("B"), ord("p"), ord("P")) or key in (KEY_LEFT, KEY_PAGE_UP):
        return "prev"
    elif key_char in (ord("d"), ord("D")) or key in (KEY_RIGHT, KEY_PAGE_DOWN):
        return "next"
    elif key_char in (13, 32):  # Enter, Space
        return "save"
    else:
        return "refresh"


def main() -> int:
    args = parse_args()
    collector_mode = bool(args.collector_jsonl)
    
    # Проверка зависимостей
    if not args.no_interactive and not HAS_CV:
        print("[error] OpenCV не установлен. Установите opencv-python или используйте --no-interactive")
        return 1
    
    valid_type_codes = load_valid_type_codes(args.valid_type_codes)

    # Проверка путей
    reader_path: Optional[Path] = None
    model_path: Optional[Path] = None
    source_path: Optional[Path] = None
    if collector_mode:
        for jsonl_path in args.collector_jsonl:
            if not jsonl_path.exists():
                print(f"[error] collector JSONL не найден: {jsonl_path}")
                return 1
    else:
        if args.reader is None or args.model is None or args.source is None:
            print("[error] без --collector-jsonl нужны --reader, --model и --source")
            return 1

        reader_path = args.reader.resolve()
        if not reader_path.exists():
            print(f"[error] read_container.py не найден: {reader_path}")
            return 1

        model_path = args.model.resolve()
        if not model_path.exists():
            print(f"[error] модель не найдена: {model_path}")
            return 1

        source_path = args.source.resolve()
        if not source_path.exists():
            print(f"[error] источник не найден: {source_path}")
            return 1
    
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_images_dir = output_dir / "images"
    dataset_path = output_dir / args.dataset_file

    if collector_mode and args.collector_pairs:
        try:
            pair_summary = run_collector_pair_builder(
                args,
                output_images_dir=output_images_dir,
                dataset_path=dataset_path,
                valid_type_codes=valid_type_codes,
            )
        except Exception as exc:
            print(f"[error] {exc}")
            return 1
        summary = {
            "reader": "",
            "model": "",
            "source": "",
            "collector_jsonl": [str(path.resolve()) for path in args.collector_jsonl],
            "output_dir": str(output_dir),
            "dataset_file": str(dataset_path),
            "copy_images": args.copy_images,
            "scenario": args.scenario_name,
            "scenario_version": args.scenario_version,
            "interactive": not args.no_interactive,
            **pair_summary,
        }
        summary_path = output_dir / "dataset_build_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\n" + "=" * 50)
        print("[done] Pair dataset generation complete")
        print(f"[done] dataset: {dataset_path}")
        print(f"[done] summary: {summary_path}")
        print(
            f"[stats] pairs={pair_summary['pairs_total']} "
            f"saved={pair_summary['saved']} "
            f"skipped={pair_summary['skipped']} "
            f"duplicates={pair_summary['duplicates']} "
            f"errors={pair_summary['errors']}"
        )
        return 0
    
    # Сбор входных samples
    if collector_mode:
        samples = collector_samples(
            [path.resolve() for path in args.collector_jsonl],
            include_unsaved_images=args.include_unsaved_collector_images,
            valid_type_codes=valid_type_codes,
        )
        if args.max_images is not None:
            samples = samples[: args.max_images]
        if not samples:
            print("[error] в collector JSONL не найдено сохраненных изображений")
            return 1
    else:
        assert source_path is not None
        image_paths = collect_images(source_path, recursive=args.recursive)
        if not image_paths:
            print(f"[error] изображения не найдены: {source_path}")
            return 1

        if args.max_images is not None:
            image_paths = image_paths[: args.max_images]
        samples = [
            {
                "image_path": image_path,
                "case_stem": image_path.stem,
                "expected": None,
                "result": None,
                "metadata": {},
            }
            for image_path in image_paths
        ]

    print(f"[info] найдено {len(samples)} изображений")
    if model_path is not None:
        print(f"[info] модель: {model_path}")
    if collector_mode:
        print(f"[info] collector JSONL: {', '.join(str(p.resolve()) for p in args.collector_jsonl)}")
        print(f"[info] valid type_size codes: {len(valid_type_codes)}")
    print(f"[info] вывод: {output_dir}")
    print(f"[info] режим: {'интерактивный' if not args.no_interactive else 'автоматический'}")
    
    if not args.no_interactive:
        print("[info] управление: Y=сохранить, N=пропустить, Q=выход")
    
    # Загрузка существующих ID
    existing_ids = load_existing_case_ids(dataset_path)
    
    processed = 0
    saved = 0
    skipped = 0
    errors = 0
    latencies_ms: List[float] = []
    saved_indices: Set[int] = set()
    
    idx = 0
    while idx < len(samples):
        sample = samples[idx]
        display_idx = idx + 1
        image_path = Path(sample["image_path"])
        print(f"\n[{display_idx}/{len(samples)}] обработка {image_path.name}...")
        
        started = time.perf_counter()
        
        if collector_mode:
            result = sample["result"]
            elapsed_ms = float(result.get("elapsed_ms", 0.0) or 0.0)
        else:
            assert reader_path is not None
            assert model_path is not None
            # Запуск read_container.py
            result = run_reader(
                reader_path=reader_path,
                model_path=model_path,
                image_path=image_path,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                merge_iou=args.merge_iou
            )

            elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        
        if result is None:
            errors += 1
            idx += 1
            print(f"[warn] нет результата для {image_path.name}")
            continue
        
        processed += 1
        
        # Парсинг результата
        expected = dict(sample["expected"]) if collector_mode else parse_reader_result(result)
        iso = build_iso_code(expected)
        
        print(f"[result] ISO: {iso or '(пусто)'}, Type/Size: {expected.get('type_size_code', '')}")
        
        # Определение действия
        should_save = False
        manual_corrected = False
        
        if args.no_interactive:
            # Автоматический режим - сохранять всё
            should_save = True
            print("[auto] автоматическое сохранение")
        else:
            # Интерактивный режим
            decision = show_preview(image_path, result, expected)
            
            if decision == "quit":
                print("[quit] выход по запросу пользователя")
                break
            elif decision == "prev":
                idx = max(0, idx - 1)
                continue
            elif decision == "next":
                idx = min(len(samples) - 1, idx + 1)
                continue
            elif decision == "refresh":
                continue
            elif decision == "edit":
                expected = edit_expected_in_console(expected, valid_type_codes)
                result["result"] = format_container_result_like_reader(expected)
                sample["expected"] = expected
                sample["result"] = result
                manual_corrected = True
                should_save = True
                print("[save] исправлено пользователем (E)")
            elif decision == "save":
                should_save = True
                print("[save] подтверждено пользователем (Y)")
            else:
                skipped += 1
                idx += 1
                print("[skip] отклонено пользователем (N)")
                continue
        
        if not should_save:
            idx += 1
            continue

        if idx in saved_indices:
            print("[skip] already saved in this session")
            idx += 1
            continue
        
        # Генерация уникального ID
        case_id = ensure_unique_case_id(str(sample.get("case_stem") or image_path.stem), existing_ids)
        existing_ids.add(case_id)
        
        # Получение количества детекций (если есть в выводе)
        raw_count = result.get("raw_detections_count", 0)
        det_count = result.get("detections_count", 0)
        duplicates_removed = result.get("duplicates_removed", 0)
        final_type_status = type_size_check(expected.get("type_size_code", ""), valid_type_codes)
        final_number_check_ok = iso6346_check_ok(build_iso_code(expected))
        final_metadata = {
            **dict(sample.get("metadata") or {}),
            "source_image": str(image_path),
            "model": str(model_path) if model_path is not None else "",
            "latency_ms": round(elapsed_ms, 3),
            "raw_detections_count": raw_count,
            "detections_count": det_count,
            "duplicates_removed": duplicates_removed,
            "manual_corrected": manual_corrected,
            "final_check_ok": final_number_check_ok,
            "final_type_size_present": final_type_status["present"],
            "final_type_size_format_ok": final_type_status["format_ok"],
            "final_type_size_allowlist_ok": final_type_status["allowlist_ok"],
            "final_type_size_ok": final_type_status["ok"],
        }
        
        # Сохранение кейса
        save_case(
            source_image=image_path,
            output_images_dir=output_images_dir,
            dataset_path=dataset_path,
            case_id=case_id,
            scenario_name=args.scenario_name,
            scenario_version=args.scenario_version,
            expected=expected,
            metadata=final_metadata,
            copy_images=args.copy_images
        )
        
        saved += 1
        saved_indices.add(idx)
        idx += 1
        print(f"[save] {case_id} | ISO: {iso or '(пусто)'}")
    
    # Статистика
    if not args.no_interactive and HAS_CV:
        cv2.destroyAllWindows()

    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    
    summary = {
        "reader": str(reader_path) if reader_path is not None else "",
        "model": str(model_path) if model_path is not None else "",
        "source": str(source_path) if source_path is not None else "",
        "collector_jsonl": [str(path.resolve()) for path in args.collector_jsonl] if collector_mode else [],
        "output_dir": str(output_dir),
        "dataset_file": str(dataset_path),
        "images_total": len(samples),
        "processed": processed,
        "saved": saved,
        "skipped": skipped,
        "errors": errors,
        "avg_latency_ms": round(avg_latency, 3),
        "copy_images": args.copy_images,
        "scenario": args.scenario_name,
        "scenario_version": args.scenario_version,
        "interactive": not args.no_interactive,
    }
    
    summary_path = output_dir / "dataset_build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print("\n" + "=" * 50)
    print("[done] Генерация датасета завершена")
    print(f"[done] датасет: {dataset_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[stats] всего изображений: {len(samples)}")
    print(f"[stats] обработано: {processed}")
    print(f"[stats] сохранено: {saved}")
    print(f"[stats] пропущено: {skipped}")
    print(f"[stats] ошибок: {errors}")
    print(f"[stats] среднее время обработки: {avg_latency:.2f} мс")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

