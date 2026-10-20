"""Collect labeled vehicle snapshots when the weighing API allows entry."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


ENTRY_ALLOWED = "Номер распознан: въезд разрешен"
SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def output_root(config: dict[str, Any]) -> Path:
    path = Path(config["output_root"])
    return path if path.is_absolute() else SCRIPT_DIR / path


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


def save_event(
    side: str,
    number: str,
    camera: dict[str, Any],
    root: Path,
    timeout_seconds: float,
    session: requests.Session,
) -> None:
    response = session.get(camera["snapshot_url"], timeout=timeout_seconds)
    response.raise_for_status()

    stem = f"{side}_{timestamp()}"
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    (images_dir / f"{stem}.jpg").write_bytes(response.content)
    (labels_dir / f"{stem}.txt").write_text(f"{number}\n", encoding="utf-8")
    print(f"Saved {camera['name']}: {stem}.jpg -> {number}")


def process_item(
    item: dict[str, Any],
    config: dict[str, Any],
    captured_sides: set[str],
    session: requests.Session,
) -> None:
    root = output_root(config)
    timeout_seconds = float(config["timeout_seconds"])
    number = str(item.get("car") or "")

    for side in ("left", "right"):
        if item.get(side) != ENTRY_ALLOWED:
            captured_sides.discard(side)
            continue

        if side in captured_sides:
            continue

        save_event(side, number, config["cameras"][side], root, timeout_seconds, session)
        captured_sides.add(side)


def poll_once(config: dict[str, Any], captured_sides: set[str], session: requests.Session) -> None:
    response = session.get(config["state_api_url"], timeout=float(config["timeout_seconds"]))
    response.raise_for_status()
    items = response.json().get("items", [])

    if items:
        process_item(items[0], config, captured_sides, session)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect labeled Axis snapshots from weighing events.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    captured_sides: set[str] = set()
    session = requests.Session()

    print("Collector started. Stop with Ctrl+C.")
    try:
        while True:
            try:
                poll_once(config, captured_sides, session)
            except requests.RequestException as error:
                print(f"Request error: {error}")
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                print(f"Response/config error: {error}")

            time.sleep(float(config["poll_interval_seconds"]))
    except KeyboardInterrupt:
        print("Collector stopped.")


if __name__ == "__main__":
    main()
