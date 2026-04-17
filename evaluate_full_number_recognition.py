#!/usr/bin/env python3
"""
Evaluate container-number recognition quality against full-number labels.

Expected ground-truth input:
  Output directory from `convert_yolo_char_labels_to_full_numbers.py`,
  containing:
    - manifest.jsonl (preferred), or
    - per-split text files (<gt_dir>/<split>/<image_stem>.txt)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_SPLITS = ("test",)
ISO_PATTERN = re.compile(r"^[A-Z]{4}[0-9]{7}$")
NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


@dataclass
class DetectionChar:
    char: str
    confidence: float
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full container-number recognition quality."
    )
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt).")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Dataset root (with train/val/test/images).",
    )
    parser.add_argument(
        "--ground-truth-dir",
        required=True,
        help="Directory produced by converter script (contains manifest.jsonl).",
    )
    parser.add_argument(
        "--classes-file",
        default=None,
        help=(
            "Optional classes.txt for class_id->character mapping. "
            "Default: <dataset_root>/train/labels/classes.txt"
        ),
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Splits to evaluate.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--device", default=None, help="Device (e.g. cpu, 0, cuda:0).")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional image limit.")
    parser.add_argument(
        "--line-y-threshold-factor",
        type=float,
        default=0.75,
        help="Line grouping threshold factor relative to median character height.",
    )
    parser.add_argument(
        "--vertical-ratio-threshold",
        type=float,
        default=1.30,
        help="If y_spread > x_spread * threshold, detections are sorted top-to-bottom.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports. Default: <ground_truth_dir>/eval_<timestamp>",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.upper())


def load_classes(classes_file: Path) -> List[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"classes file not found: {classes_file}")
    classes = [line.strip().upper() for line in classes_file.read_text(encoding="utf-8").splitlines()]
    classes = [c for c in classes if c != ""]
    if not classes:
        raise ValueError(f"classes file is empty: {classes_file}")
    return classes


def find_image_for_stem(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    candidates = sorted(p for p in images_dir.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return candidates[0] if candidates else None


def load_ground_truth_records(
    dataset_root: Path,
    ground_truth_dir: Path,
    splits: Sequence[str],
) -> List[Dict]:
    manifest_path = ground_truth_dir / "manifest.jsonl"
    records: List[Dict] = []
    split_set = {s.strip() for s in splits if s.strip()}

    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            split = str(row.get("split", ""))
            if split not in split_set:
                continue
            full_number = normalize_text(str(row.get("full_number", "")))
            image_rel = row.get("image_rel")
            if image_rel:
                image_path = dataset_root / image_rel
            else:
                # Fallback from output label filename.
                output_label_rel = row.get("output_label_rel")
                if output_label_rel:
                    stem = Path(output_label_rel).stem
                    image_path = find_image_for_stem(dataset_root / split / "images", stem)
                else:
                    image_path = None

            records.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "image_rel": str(image_path.relative_to(dataset_root).as_posix()) if image_path else None,
                    "gt_text": full_number,
                }
            )
        return records

    # Fallback mode: read <gt_dir>/<split>/*.txt
    for split in split_set:
        gt_split_dir = ground_truth_dir / split
        images_dir = dataset_root / split / "images"
        if not gt_split_dir.exists():
            continue

        for gt_file in sorted(gt_split_dir.glob("*.txt")):
            gt_text = normalize_text(gt_file.read_text(encoding="utf-8").strip())
            image_path = find_image_for_stem(images_dir=images_dir, stem=gt_file.stem)
            records.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "image_rel": str(image_path.relative_to(dataset_root).as_posix()) if image_path else None,
                    "gt_text": gt_text,
                }
            )

    return records


def iso6346_char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)

    value = 10
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        while value in (11, 22, 33):
            value += 1
        if ch == letter:
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


def group_lines(
    detections_by_y: List[DetectionChar],
    y_threshold: float,
) -> List[List[DetectionChar]]:
    lines: List[List[DetectionChar]] = []
    line_centers: List[float] = []

    for det in detections_by_y:
        placed = False
        for idx, center in enumerate(line_centers):
            if abs(det.y_center - center) <= y_threshold:
                lines[idx].append(det)
                line_centers[idx] = statistics.mean(x.y_center for x in lines[idx])
                placed = True
                break
        if not placed:
            lines.append([det])
            line_centers.append(det.y_center)

    paired = sorted(zip(lines, line_centers), key=lambda t: t[1])
    return [line for line, _ in paired]


def sort_detections_for_reading(
    detections: List[DetectionChar],
    line_y_threshold_factor: float,
    vertical_ratio_threshold: float,
) -> Tuple[List[DetectionChar], str]:
    if len(detections) <= 1:
        return detections, "single"

    xs = [d.x_center for d in detections]
    ys = [d.y_center for d in detections]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)

    if spread_x <= 1e-9 and spread_y > 0:
        return sorted(detections, key=lambda d: d.y_center), "vertical"

    if spread_y > spread_x * vertical_ratio_threshold:
        return sorted(detections, key=lambda d: d.y_center), "vertical"

    median_h = statistics.median(d.height for d in detections)
    y_threshold = max(median_h * line_y_threshold_factor, 1.0)
    by_y = sorted(detections, key=lambda d: (d.y_center, d.x_center))
    lines = group_lines(by_y, y_threshold=y_threshold)

    if len(lines) == 1:
        return sorted(lines[0], key=lambda d: d.x_center), "horizontal"

    output: List[DetectionChar] = []
    for line in lines:
        output.extend(sorted(line, key=lambda d: d.x_center))
    return output, "multi_line"


def lcs_length(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def levenshtein_distance(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev, curr = curr, prev
    return prev[m]


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_char_from_class_id(
    class_id: int,
    class_map: Optional[Sequence[str]],
    model_names: Dict[int, str],
) -> str:
    if class_map and 0 <= class_id < len(class_map):
        return normalize_text(class_map[class_id])

    raw = model_names.get(class_id, str(class_id))
    return normalize_text(str(raw))


def main() -> int:
    args = parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    ground_truth_dir = Path(args.ground_truth_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (ground_truth_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    )
    ensure_dir(output_dir)

    classes_file = (
        Path(args.classes_file).resolve()
        if args.classes_file
        else (dataset_root / "train" / "labels" / "classes.txt").resolve()
    )
    class_map: Optional[List[str]] = None
    if classes_file.exists():
        class_map = load_classes(classes_file)
        print(f"[info] class map loaded: {classes_file} ({len(class_map)} classes)")
    else:
        print(f"[warn] classes file not found, will rely on model.names: {classes_file}")

    records = load_ground_truth_records(
        dataset_root=dataset_root,
        ground_truth_dir=ground_truth_dir,
        splits=args.splits,
    )

    records = [r for r in records if r.get("gt_text")]
    if args.max_images is not None:
        records = records[: args.max_images]

    if not records:
        raise RuntimeError("No ground-truth records found for selected splits.")

    # Import ultralytics lazily to allow --help without dependency errors.
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Package 'ultralytics' is not installed in current Python environment. "
            "Activate your project venv and install it, e.g. 'pip install ultralytics'."
        ) from exc

    model = YOLO(str(Path(args.model).resolve()))
    model_names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    per_image_path = output_dir / "per_image.jsonl"
    summary_path = output_dir / "summary.json"

    total = 0
    skipped_no_image = 0
    inference_errors = 0
    exact_matches = 0
    iso_gt_valid_count = 0
    iso_pred_valid_count = 0
    iso_pred_and_exact_count = 0
    lcs_tp_total = 0
    char_fp_total = 0
    char_fn_total = 0
    edit_distances: List[int] = []
    similarities: List[float] = []
    inference_times_ms: List[float] = []
    unmatched_label_count = 0

    with per_image_path.open("w", encoding="utf-8") as pf:
        for idx, row in enumerate(records, start=1):
            image_path: Optional[Path] = row.get("image_path")
            split = row.get("split")
            gt_text = normalize_text(row.get("gt_text", ""))

            if not image_path or not image_path.exists():
                skipped_no_image += 1
                continue

            total += 1
            start = time.perf_counter()
            try:
                predictions = model.predict(
                    source=str(image_path),
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    max_det=args.max_det,
                    verbose=False,
                )
                result = predictions[0]
            except Exception as exc:
                inference_errors += 1
                pf.write(
                    json.dumps(
                        {
                            "split": split,
                            "image_rel": image_path.relative_to(dataset_root).as_posix(),
                            "gt_text": gt_text,
                            "pred_text": "",
                            "exact_match": False,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            inference_times_ms.append(elapsed_ms)

            detections: List[DetectionChar] = []
            if getattr(result, "boxes", None) is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                classes = result.boxes.cls.cpu().tolist()

                for box, conf, cls in zip(xyxy, confs, classes):
                    class_id = int(cls)
                    char = build_char_from_class_id(
                        class_id=class_id,
                        class_map=class_map,
                        model_names=model_names,
                    )
                    if len(char) != 1:
                        unmatched_label_count += 1
                        # Keep only unambiguous 1-char labels for full-number reconstruction.
                        continue

                    x1, y1, x2, y2 = [float(v) for v in box]
                    detections.append(
                        DetectionChar(
                            char=char,
                            confidence=float(conf),
                            x_center=(x1 + x2) / 2.0,
                            y_center=(y1 + y2) / 2.0,
                            width=max(0.0, x2 - x1),
                            height=max(0.0, y2 - y1),
                        )
                    )

            sorted_detections, sort_mode = sort_detections_for_reading(
                detections=detections,
                line_y_threshold_factor=args.line_y_threshold_factor,
                vertical_ratio_threshold=args.vertical_ratio_threshold,
            )
            pred_text = "".join(det.char for det in sorted_detections)
            pred_text = normalize_text(pred_text)

            exact_match = pred_text == gt_text
            if exact_match:
                exact_matches += 1

            lcs_tp = lcs_length(gt_text, pred_text)
            char_fp = max(0, len(pred_text) - lcs_tp)
            char_fn = max(0, len(gt_text) - lcs_tp)
            lcs_tp_total += lcs_tp
            char_fp_total += char_fp
            char_fn_total += char_fn

            edit_distance = levenshtein_distance(gt_text, pred_text)
            denom = max(len(gt_text), len(pred_text), 1)
            normalized_similarity = 1.0 - (edit_distance / denom)
            edit_distances.append(edit_distance)
            similarities.append(normalized_similarity)

            gt_iso_valid = is_iso6346_valid(gt_text)
            pred_iso_valid = is_iso6346_valid(pred_text)
            if gt_iso_valid:
                iso_gt_valid_count += 1
            if pred_iso_valid:
                iso_pred_valid_count += 1
            if pred_iso_valid and exact_match:
                iso_pred_and_exact_count += 1

            per_image_record = {
                "split": split,
                "image_rel": image_path.relative_to(dataset_root).as_posix(),
                "gt_text": gt_text,
                "pred_text": pred_text,
                "exact_match": exact_match,
                "sort_mode": sort_mode,
                "detections_count": len(sorted_detections),
                "edit_distance": edit_distance,
                "normalized_similarity": round(normalized_similarity, 6),
                "char_tp_lcs": lcs_tp,
                "char_fp": char_fp,
                "char_fn": char_fn,
                "gt_iso6346_valid": gt_iso_valid,
                "pred_iso6346_valid": pred_iso_valid,
                "inference_time_ms": round(elapsed_ms, 3),
            }
            pf.write(json.dumps(per_image_record, ensure_ascii=False) + "\n")

            if idx % 100 == 0:
                print(f"[info] processed {idx}/{len(records)}")

    accuracy = safe_div(exact_matches, total)
    precision = safe_div(lcs_tp_total, lcs_tp_total + char_fp_total)
    recall = safe_div(lcs_tp_total, lcs_tp_total + char_fn_total)
    f1 = safe_div(2.0 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    summary = {
        "model_path": str(Path(args.model).resolve()),
        "dataset_root": str(dataset_root),
        "ground_truth_dir": str(ground_truth_dir),
        "splits": list(args.splits),
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "max_det": args.max_det,
        },
        "counts": {
            "records_loaded": len(records),
            "evaluated": total,
            "skipped_no_image": skipped_no_image,
            "inference_errors": inference_errors,
            "unmatched_labels_filtered": unmatched_label_count,
        },
        "metrics": {
            # Binary full-number metric: exact match (correct / incorrect).
            "accuracy": round(accuracy, 6),
            "exact_match_rate": round(accuracy, 6),
            "exact_matches": exact_matches,
            "incorrect_matches": max(0, total - exact_matches),
            # Requested precision/recall are reported on character level (LCS-based),
            # because they are informative even when exact-match is binary.
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            # Extra metrics.
            "avg_edit_distance": round(statistics.mean(edit_distances), 6) if edit_distances else 0.0,
            "avg_normalized_similarity": round(statistics.mean(similarities), 6) if similarities else 0.0,
            "iso6346_gt_valid_rate": round(safe_div(iso_gt_valid_count, total), 6),
            "iso6346_pred_valid_rate": round(safe_div(iso_pred_valid_count, total), 6),
            "iso6346_pred_valid_and_exact_rate": round(safe_div(iso_pred_and_exact_count, total), 6),
            "avg_inference_time_ms": round(statistics.mean(inference_times_ms), 6) if inference_times_ms else 0.0,
            "p95_inference_time_ms": (
                round(statistics.quantiles(inference_times_ms, n=20)[18], 6)
                if len(inference_times_ms) >= 20
                else (round(max(inference_times_ms), 6) if inference_times_ms else 0.0)
            ),
        },
        "outputs": {
            "per_image_jsonl": str(per_image_path),
            "summary_json": str(summary_path),
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] Evaluation completed.")
    print(f"[done] Per-image: {per_image_path}")
    print(f"[done] Summary:   {summary_path}")
    print("")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
