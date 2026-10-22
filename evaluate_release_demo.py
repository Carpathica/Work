"""Evaluate release-demo plate recognition against paired dataset labels."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LOOK_ALIKES = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
})


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def normalize_plate(value: Any) -> str:
    raw = str(value or "").upper().translate(LOOK_ALIKES)
    return "".join(character for character in raw if character.isalnum())


def extract_plate(value: Any, label: str) -> str:
    if isinstance(value, list):
        for item in value:
            result = extract_plate(item, label)
            if result:
                return result
        return ""

    if not isinstance(value, dict):
        return ""

    if value.get("label") == label:
        return str(value.get("text") or "")

    direct = value.get(label)
    if isinstance(direct, (str, int, float)):
        return str(direct)

    for key in ("result", "items", "data"):
        result = extract_plate(value.get(key), label)
        if result:
            return result
    return ""


def extract_yolo_plate(value: Any) -> str:
    if not isinstance(value, dict):
        return ""

    result = value.get("result", value)
    if not isinstance(result, dict):
        return ""

    items = result.get("items")
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return str(items[0].get("primary_number") or "")
    return str(result.get("primary_number") or "")


def login(session: requests.Session, config: dict[str, Any]) -> str:
    base_url = str(config["base_url"]).rstrip("/")
    response = session.post(
        f"{base_url}/auth/login",
        json={"username": config["username"], "password": config["password"]},
        timeout=float(config["timeout_seconds"]),
    )
    response.raise_for_status()
    return response.json()["access_token"]


def recognize_image(
    session: requests.Session,
    config: dict[str, Any],
    token: str,
    scenario: str,
    mode: str,
    image_path: Path,
) -> tuple[int, str]:
    base_url = str(config["base_url"]).rstrip("/")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    with image_path.open("rb") as image_file:
        endpoint = "/api/v1/recognize-multipart" if mode == "vlm" else "/api/v1/yolo-recognize-by-scenario"
        form_data: dict[str, str]
        if mode == "vlm":
            form_data = {"metadata": json.dumps({"scenario": scenario}, ensure_ascii=False)}
        else:
            form_data = {"scenario": scenario}
        response = session.post(
            f"{base_url}{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            data=form_data,
            files={"files": (image_path.name, image_file, mime_type)},
            timeout=float(config["timeout_seconds"]),
        )
    response.raise_for_status()
    payload = response.json()
    if mode == "yolo":
        return response.status_code, extract_yolo_plate(payload)
    return response.status_code, extract_plate(payload, str(config["plate_response_label"]))


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image", "label", "expected", "predicted", "expected_normalized",
        "predicted_normalized", "recognized", "match", "http_status", "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare release-demo recognition with plate labels.")
    parser.add_argument("--scenario", required=True, help="Name of the active server scenario.")
    parser.add_argument("--mode", choices=("vlm", "yolo"), default="vlm", help="Recognition endpoint to use.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "evaluate_release_demo_config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    dataset_root = resolve_path(str(config["dataset_root"]), config_path)
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    report_dir = resolve_path(str(config["report_dir"]), config_path)
    rows: list[dict[str, Any]] = []

    session = requests.Session()
    token = login(session, config)

    for image_path in iter_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        expected = ""
        predicted = ""
        status = ""
        error = ""

        if not label_path.exists():
            error = "missing_label"
        else:
            expected = label_path.read_text(encoding="utf-8").strip()
            try:
                status, predicted = recognize_image(session, config, token, args.scenario, args.mode, image_path)
            except requests.RequestException as request_error:
                error = type(request_error).__name__

        expected_normalized = normalize_plate(expected)
        predicted_normalized = normalize_plate(predicted)
        recognized = bool(predicted_normalized)
        match = bool(expected_normalized and expected_normalized == predicted_normalized)
        rows.append(
            {
                "image": image_path.name,
                "label": label_path.name,
                "expected": expected,
                "predicted": predicted,
                "expected_normalized": expected_normalized,
                "predicted_normalized": predicted_normalized,
                "recognized": recognized,
                "match": match,
                "http_status": status,
                "error": error,
            }
        )
        print(f"{image_path.name}: expected={expected or '-'} predicted={predicted or '-'} match={match}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = report_dir / f"release_demo_plate_evaluation_{timestamp}.csv"
    write_report(report_path, rows)

    total = len(rows)
    recognized_count = sum(row["recognized"] for row in rows)
    matches = sum(row["match"] for row in rows)
    errors = sum(bool(row["error"]) for row in rows)
    accuracy = matches / total if total else 0.0
    print(f"\nTotal: {total}")
    print(f"Recognized: {recognized_count}; not recognized: {total - recognized_count}")
    print(f"Correct: {matches}; accuracy: {accuracy:.2%}; errors: {errors}")
    print(f"Mode: {args.mode}; scenario: {args.scenario}")
    print(f"CSV report: {report_path}")


if __name__ == "__main__":
    main()
