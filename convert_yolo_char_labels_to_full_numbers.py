#!/usr/bin/env python3
"""
Convert per-character YOLO labels into full container-number labels.

Input (YOLO labels):
  <class_id> <x_center> <y_center> <width> <height>

Output:
  <output_dir>/
    manifest.jsonl
    summary.json
    train/<image_stem>.txt
    val/<image_stem>.txt
    test/<image_stem>.txt
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_SPLITS = ("train", "val", "test")
ISO_PATTERN = re.compile(r"^[A-Z]{4}[0-9]{7}$")


@dataclass
class CharAnnotation:
    class_id: int
    char: str
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO per-character labels into full-number labels."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root with train/val/test subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for converted full labels.",
    )
    parser.add_argument(
        "--classes-file",
        default=None,
        help=(
            "Path to classes.txt (one class name per line). "
            "Default: <dataset_root>/train/labels/classes.txt"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to process (e.g. test or train val test).",
    )
    parser.add_argument(
        "--line-y-threshold-factor",
        type=float,
        default=0.75,
        help=(
            "Line grouping threshold factor relative to median character height. "
            "Used for ordering multi-line labels."
        ),
    )
    parser.add_argument(
        "--vertical-ratio-threshold",
        type=float,
        default=1.30,
        help=(
            "If y_spread > x_spread * threshold, annotations are treated as vertical and "
            "sorted top-to-bottom."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for number of label files per split (for quick tests).",
    )
    return parser.parse_args()


def load_classes(classes_file: Path) -> List[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"classes file not found: {classes_file}")

    classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()]
    classes = [c for c in classes if c != ""]
    if not classes:
        raise ValueError(f"classes file is empty: {classes_file}")
    return classes


def parse_yolo_label_file(label_file: Path, classes: Sequence[str]) -> Tuple[List[CharAnnotation], List[str]]:
    annotations: List[CharAnnotation] = []
    warnings: List[str] = []

    for line_idx, raw_line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            warnings.append(f"{label_file.name}:{line_idx} invalid format: '{line}'")
            continue

        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            warnings.append(f"{label_file.name}:{line_idx} parse error: '{line}'")
            continue

        if class_id < 0 or class_id >= len(classes):
            char = "?"
            warnings.append(
                f"{label_file.name}:{line_idx} class_id={class_id} is out of classes range [0..{len(classes)-1}]"
            )
        else:
            char = classes[class_id].strip().upper()
            if char == "":
                char = "?"
                warnings.append(f"{label_file.name}:{line_idx} empty class name for id={class_id}")

        annotations.append(
            CharAnnotation(
                class_id=class_id,
                char=char,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )

    return annotations, warnings


def group_lines(
    sorted_by_y: List[CharAnnotation],
    y_threshold: float,
) -> List[List[CharAnnotation]]:
    lines: List[List[CharAnnotation]] = []
    line_centers: List[float] = []

    for ann in sorted_by_y:
        placed = False
        for idx, center in enumerate(line_centers):
            if abs(ann.y_center - center) <= y_threshold:
                lines[idx].append(ann)
                line_centers[idx] = statistics.mean(item.y_center for item in lines[idx])
                placed = True
                break
        if not placed:
            lines.append([ann])
            line_centers.append(ann.y_center)

    line_pairs = list(zip(lines, line_centers))
    line_pairs.sort(key=lambda item: item[1])
    return [line for line, _ in line_pairs]


def sort_annotations_for_reading(
    annotations: List[CharAnnotation],
    line_y_threshold_factor: float,
    vertical_ratio_threshold: float,
) -> Tuple[List[CharAnnotation], str]:
    if len(annotations) <= 1:
        return annotations, "single"

    xs = [ann.x_center for ann in annotations]
    ys = [ann.y_center for ann in annotations]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)

    if spread_x <= 1e-9 and spread_y > 0:
        return sorted(annotations, key=lambda ann: ann.y_center), "vertical"

    if spread_y > spread_x * vertical_ratio_threshold:
        return sorted(annotations, key=lambda ann: ann.y_center), "vertical"

    median_h = statistics.median(ann.height for ann in annotations)
    y_threshold = max(median_h * line_y_threshold_factor, 0.01)

    sorted_by_y = sorted(annotations, key=lambda ann: (ann.y_center, ann.x_center))
    lines = group_lines(sorted_by_y, y_threshold=y_threshold)

    if len(lines) == 1:
        return sorted(lines[0], key=lambda ann: ann.x_center), "horizontal"

    output: List[CharAnnotation] = []
    for line in lines:
        output.extend(sorted(line, key=lambda ann: ann.x_center))
    return output, "multi_line"


def find_image_for_label(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    candidates = sorted(p for p in images_dir.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if candidates:
        return candidates[0]
    return None


def iso6346_char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)

    # ISO 6346 letter values: A=10, B=12, C=13, ... skip 11,22,33.
    value = 10
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        while value in (11, 22, 33):
            value += 1
        if letter == ch:
            return value
        value += 1
    raise ValueError(f"Unsupported ISO letter: {ch}")


def iso6346_check_digit(code10: str) -> int:
    total = 0
    for pos, ch in enumerate(code10):
        total += iso6346_char_value(ch) * (1 << pos)
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def is_iso6346_valid(code: str) -> bool:
    if not ISO_PATTERN.match(code):
        return False
    expected = iso6346_check_digit(code[:10])
    return expected == int(code[10])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    classes_file = (
        Path(args.classes_file).resolve()
        if args.classes_file
        else (dataset_root / "train" / "labels" / "classes.txt").resolve()
    )
    classes = load_classes(classes_file)

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    all_records: List[Dict] = []
    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "classes_file": str(classes_file),
        "classes_count": len(classes),
        "splits": {},
        "total_files_processed": 0,
        "total_warnings": 0,
    }

    for split in args.splits:
        split_name = split.strip()
        if not split_name:
            continue

        labels_dir = dataset_root / split_name / "labels"
        images_dir = dataset_root / split_name / "images"
        out_split_dir = output_dir / split_name
        ensure_dir(out_split_dir)

        if not labels_dir.exists():
            print(f"[warn] labels dir does not exist, skip split '{split_name}': {labels_dir}")
            continue

        label_files = sorted(
            p
            for p in labels_dir.glob("*.txt")
            if p.name.lower() != "classes.txt"
        )
        if args.max_files is not None:
            label_files = label_files[: args.max_files]

        split_count = 0
        split_warning_count = 0
        split_iso_valid_count = 0
        split_missing_image_count = 0

        for label_file in label_files:
            split_count += 1
            annotations, warnings = parse_yolo_label_file(label_file, classes=classes)
            split_warning_count += len(warnings)

            sorted_annotations, sort_mode = sort_annotations_for_reading(
                annotations=annotations,
                line_y_threshold_factor=args.line_y_threshold_factor,
                vertical_ratio_threshold=args.vertical_ratio_threshold,
            )
            full_number = "".join(ann.char for ann in sorted_annotations)
            iso_valid = is_iso6346_valid(full_number)
            if iso_valid:
                split_iso_valid_count += 1

            out_label_file = out_split_dir / f"{label_file.stem}.txt"
            out_label_file.write_text(full_number + "\n", encoding="utf-8")

            image_path = find_image_for_label(images_dir=images_dir, stem=label_file.stem)
            if image_path is None:
                split_missing_image_count += 1

            record = {
                "split": split_name,
                "label_rel": label_file.relative_to(dataset_root).as_posix(),
                "image_rel": image_path.relative_to(dataset_root).as_posix() if image_path else None,
                "output_label_rel": out_label_file.relative_to(output_dir).as_posix(),
                "full_number": full_number,
                "char_count": len(sorted_annotations),
                "iso6346_valid": iso_valid,
                "sort_mode": sort_mode,
                "warnings": warnings,
            }
            all_records.append(record)

        summary["splits"][split_name] = {
            "files_processed": split_count,
            "warnings": split_warning_count,
            "iso6346_valid_count": split_iso_valid_count,
            "missing_images": split_missing_image_count,
        }
        summary["total_files_processed"] += split_count
        summary["total_warnings"] += split_warning_count

        print(
            f"[info] split={split_name}: files={split_count}, warnings={split_warning_count}, "
            f"iso_valid={split_iso_valid_count}, missing_images={split_missing_image_count}"
        )

    with manifest_path.open("w", encoding="utf-8") as mf:
        for record in all_records:
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] Manifest: {manifest_path}")
    print(f"[done] Summary:  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
