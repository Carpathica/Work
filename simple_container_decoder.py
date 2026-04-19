#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Detection:
    idx: int
    char: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass
class IsoCandidate:
    iso_container: str
    detected_check_digit: Optional[str]
    calculated_check_digit: Optional[int]
    check_digit_match: Optional[bool]
    used_indices: List[int]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple coordinate-based container decoder: "
            "YOLO -> sorting by coordinates -> ISO + check digit + additional code."
        )
    )
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model.")
    parser.add_argument("--source", required=True, help="Image file, directory, or glob.")
    parser.add_argument("--output", default="simple_decode_results.jsonl", help="JSONL output path.")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--device", default=None, help="Device, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--recursive", action="store_true", help="Recursive image search in directory.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional image limit.")
    return parser.parse_args()


def collect_images(source: str, recursive: bool) -> List[Path]:
    src = Path(source)
    if src.exists() and src.is_file():
        return [src.resolve()]

    if src.exists() and src.is_dir():
        pattern = "**/*" if recursive else "*"
        return sorted(
            p.resolve()
            for p in src.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    root = Path(".").resolve()
    return sorted(
        p.resolve()
        for p in root.glob(source)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_letter_values() -> Dict[str, int]:
    out: Dict[str, int] = {}
    value = 10
    skip = {11, 22, 33}
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        while value in skip:
            value += 1
        out[ch] = value
        value += 1
    return out


LETTER_VALUES = build_letter_values()


def iso_char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    return LETTER_VALUES[ch]


def iso_check_digit(code10: str) -> int:
    total = 0
    for pos, ch in enumerate(code10):
        total += iso_char_value(ch) * (1 << pos)
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def point_distance(a: Detection, b: Detection) -> float:
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / norm


def build_axes(dets: Sequence[Detection]) -> Tuple[np.ndarray, np.ndarray]:
    if len(dets) < 2:
        u = np.array([1.0, 0.0], dtype=np.float64)
        v = np.array([0.0, 1.0], dtype=np.float64)
        return u, v

    pts = np.array([(d.cx, d.cy) for d in dets], dtype=np.float64)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    u = normalize(eigvecs[:, int(np.argmax(eigvals))])
    v = normalize(np.array([-u[1], u[0]], dtype=np.float64))
    return u, v


def split_order_into_segments(
    order: Sequence[int],
    dets: Sequence[Detection],
    gap_threshold: float,
) -> List[List[int]]:
    if not order:
        return []

    ordered = list(order)
    if len(ordered) == 1:
        return [ordered]

    gaps = [point_distance(dets[ordered[i]], dets[ordered[i + 1]]) for i in range(len(ordered) - 1)]

    segments: List[List[int]] = [[ordered[0]]]
    for i, gap in enumerate(gaps):
        if gap > gap_threshold:
            segments.append([])
        segments[-1].append(ordered[i + 1])
    return segments


def normalize_segment_order(segment: Sequence[int], dets: Sequence[Detection]) -> List[int]:
    seg = list(segment)
    if len(seg) <= 1:
        return seg

    xs = [dets[i].cx for i in seg]
    ys = [dets[i].cy for i in seg]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)

    if spread_y > spread_x:
        return sorted(seg, key=lambda i: (dets[i].cy, dets[i].cx))
    return sorted(seg, key=lambda i: (dets[i].cx, dets[i].cy))


def order_by_mode(dets: Sequence[Detection], mode: str) -> List[int]:
    if mode == "x":
        return sorted(range(len(dets)), key=lambda i: dets[i].cx)
    if mode == "y":
        return sorted(range(len(dets)), key=lambda i: dets[i].cy)
    if mode == "pca":
        u_axis, _ = build_axes(dets)
        pts = np.array([(d.cx, d.cy) for d in dets], dtype=np.float64)
        proj_t = pts @ u_axis
        return sorted(range(len(dets)), key=lambda i: proj_t[i])
    raise ValueError(f"Unknown order mode: {mode}")


def build_ordered_segments(dets: Sequence[Detection]) -> List[Tuple[str, List[List[int]]]]:
    if not dets:
        return []

    out: List[Tuple[str, List[List[int]]]] = []
    char_size = statistics.median(max(d.w, d.h) for d in dets)

    for mode in ("x", "y", "pca"):
        order = order_by_mode(dets, mode)
        if len(order) <= 1:
            out.append((mode, [order]))
            continue

        gaps = [point_distance(dets[order[i]], dets[order[i + 1]]) for i in range(len(order) - 1)]
        median_gap = statistics.median(gaps)
        gap_threshold = max(median_gap * 1.8, char_size * 1.8)
        raw_segments = split_order_into_segments(order, dets, gap_threshold)
        segments = [normalize_segment_order(seg, dets) for seg in raw_segments]
        out.append((mode, segments))

    return out


def score_iso10(chars10: Sequence[str]) -> float:
    if len(chars10) != 10:
        return float("-inf")
    letters = sum(1 for c in chars10[:4] if c.isalpha())
    digits = sum(1 for c in chars10[4:] if c.isdigit())
    return letters * 2.0 + digits * 1.8


def evaluate_iso_windows(sequence: Sequence[int], dets: Sequence[Detection]) -> List[IsoCandidate]:
    out: List[IsoCandidate] = []
    if len(sequence) < 10:
        return out

    for start in range(0, len(sequence) - 9):
        base_idx = list(sequence[start : start + 10])
        base_chars = [dets[i].char for i in base_idx]
        if not all(ch.isalnum() for ch in base_chars):
            continue
        if not all(c.isalpha() for c in base_chars[:4]):
            continue
        if base_chars[3] not in {"U", "J", "Z"}:
            continue
        if not all(c.isdigit() for c in base_chars[4:]):
            continue

        base10 = "".join(base_chars)
        try:
            calc = iso_check_digit(base10)
        except Exception:
            continue

        conf_bonus = statistics.mean(dets[i].conf for i in base_idx)
        score = score_iso10(base_chars) + conf_bonus

        detected_check: Optional[str] = None
        check_match: Optional[bool] = None
        used = list(base_idx)

        if start + 10 < len(sequence):
            check_idx = sequence[start + 10]
            ch = dets[check_idx].char
            if ch.isdigit():
                detected_check = ch
                used.append(check_idx)
                check_match = int(ch) == calc
                score += 1.2 if check_match else -0.4
                iso_code = base10 + ch
            else:
                iso_code = base10 + str(calc)
        else:
            iso_code = base10 + str(calc)

        out.append(
            IsoCandidate(
                iso_container=iso_code,
                detected_check_digit=detected_check,
                calculated_check_digit=calc,
                check_digit_match=check_match,
                used_indices=used,
                score=score,
            )
        )
    return out


def choose_iso(segments: Sequence[Sequence[int]], dets: Sequence[Detection]) -> Optional[IsoCandidate]:
    best: Optional[IsoCandidate] = None
    sequences: List[List[int]] = []

    for seg in segments:
        if seg:
            sequences.append(list(seg))
    for i in range(len(segments) - 1):
        merged2 = list(segments[i]) + list(segments[i + 1])
        if merged2:
            sequences.append(merged2)
    for i in range(len(segments) - 2):
        merged3 = list(segments[i]) + list(segments[i + 1]) + list(segments[i + 2])
        if merged3:
            sequences.append(merged3)

    for seq_base in sequences:
        if len(seq_base) < 10:
            continue
        for seq in (seq_base, list(reversed(seq_base))):
            for candidate in evaluate_iso_windows(seq, dets):
                if best is None or candidate.score > best.score:
                    best = candidate
    return best


def choose_additional_code(
    segments: Sequence[Sequence[int]],
    dets: Sequence[Detection],
    used_indices: Sequence[int],
) -> Optional[Tuple[str, float]]:
    used = set(used_indices)
    best_code: Optional[str] = None
    best_score = float("-inf")

    available_segments: List[List[int]] = []
    for segment in segments:
        if any(i in used for i in segment):
            continue
        available_segments.append(list(segment))

    sequences: List[List[int]] = []
    for seg in available_segments:
        sequences.append(seg)
    for i in range(len(available_segments) - 1):
        merged = available_segments[i] + available_segments[i + 1]
        sequences.append(merged)

    for seq_raw in sequences:
        if len(seq_raw) < 4:
            continue

        for orientation, seq in (("forward", list(seq_raw)), ("reverse", list(reversed(seq_raw)))):
            for start in range(0, len(seq) - 3):
                win = seq[start : start + 4]
                chars = [dets[i].char for i in win]
                digit_count = sum(1 for c in chars if c.isdigit())
                letter_count = sum(1 for c in chars if c.isalpha())
                if digit_count != 3 or letter_count != 1:
                    continue

                # Preferred container additional-code layout (e.g. 45G1): DDLD.
                pattern_bonus = 0.0
                if chars[0].isdigit() and chars[1].isdigit() and chars[2].isalpha() and chars[3].isdigit():
                    pattern_bonus = 1.5
                elif chars[0].isdigit() and chars[1].isalpha() and chars[2].isdigit() and chars[3].isdigit():
                    pattern_bonus = 0.5
                else:
                    pattern_bonus = 0.2

                orientation_bonus = 0.1 if orientation == "forward" else 0.0
                score = statistics.mean(dets[i].conf for i in win) + pattern_bonus + orientation_bonus
                if score > best_score:
                    best_score = score
                    best_code = "".join(chars)
    if best_code is None:
        return None
    return best_code, best_score


def decode_detections(dets: Sequence[Detection]) -> Dict[str, object]:
    if not dets:
        return {
            "iso_container": None,
            "additional_code": None,
            "detected_check_digit": None,
            "calculated_check_digit": None,
            "check_digit_match": None,
        }

    layouts = build_ordered_segments(dets)
    best_iso: Optional[IsoCandidate] = None
    best_segments: Optional[List[List[int]]] = None

    for _, segments in layouts:
        iso = choose_iso(segments, dets)
        if iso is None:
            continue
        if best_iso is None or iso.score > best_iso.score:
            best_iso = iso
            best_segments = segments

    iso = best_iso
    if iso is None:
        return {
            "iso_container": None,
            "additional_code": None,
            "detected_check_digit": None,
            "calculated_check_digit": None,
            "check_digit_match": None,
        }

    additional_best: Optional[str] = None
    additional_best_score = float("-inf")
    for _, segments in layouts:
        extra = choose_additional_code(segments, dets, iso.used_indices)
        if extra is None:
            continue
        code, score = extra
        if score > additional_best_score:
            additional_best_score = score
            additional_best = code

    return {
        "iso_container": iso.iso_container,
        "additional_code": additional_best,
        "detected_check_digit": iso.detected_check_digit,
        "calculated_check_digit": iso.calculated_check_digit,
        "check_digit_match": iso.check_digit_match,
    }


def detections_from_prediction(prediction, names_map: Dict[int, str]) -> List[Detection]:
    boxes = getattr(prediction, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()

    out: List[Detection] = []
    for idx, (box, conf, cls) in enumerate(zip(xyxy, confs, classes)):
        class_id = int(cls)
        char = names_map.get(class_id, str(class_id)).strip().upper()
        if len(char) != 1 or not char.isalnum():
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        out.append(
            Detection(
                idx=idx,
                char=char,
                conf=float(conf),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return out


def main() -> int:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is not installed in current environment.") from exc

    images = collect_images(args.source, recursive=args.recursive)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise RuntimeError(f"No images found: {args.source}")

    model = YOLO(str(Path(args.model).resolve()))
    names_map = (
        {int(k): str(v) for k, v in model.names.items()}
        if isinstance(model.names, dict)
        else {i: str(n) for i, n in enumerate(model.names)}
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        for i, image_path in enumerate(images, start=1):
            prediction = model.predict(
                source=str(image_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                device=args.device,
                verbose=False,
            )[0]
            dets = detections_from_prediction(prediction, names_map)
            decoded = decode_detections(dets)
            row = {
                "image": str(image_path),
                **decoded,
                "detections_count": len(dets),
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

            if i % 20 == 0:
                print(f"[info] processed {i}/{len(images)}")

    print(f"[done] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
