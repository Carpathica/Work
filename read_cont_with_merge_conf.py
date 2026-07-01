from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

from read_container import (
    DEFAULT_MERGE_IOU,
    IMAGE_SUFFIXES,
    CameraRead,
    predict_camera_read,
)


# ISO 6346 number structure:
# 3 owner letters + equipment category (U/J/Z) + 6 serial digits + 1 check digit.
ISO_NUMBER_RE = re.compile(r"^[A-Z]{3}[UJZ][0-9]{7}$")
SIZE_TYPE_4_RE = re.compile(r"^[0-9]{2}[A-Z0-9][0-9]$")
SIZE_TYPE_4_LETTERS_RE = re.compile(r"^[0-9]{2}[A-Z]{1,2}$")


class PositionChoice(NamedTuple):
    index: int
    char: str
    conf: float
    source: str
    cam1: tuple[str, float] | None
    cam2: tuple[str, float] | None


class ProbabilisticMergeRead(NamedTuple):
    primary_number: str
    check_ok: bool
    size_type_code: str | None
    fusion: str
    camera1: CameraRead
    camera2: CameraRead
    choices: list[PositionChoice]


def _build_iso_letter_values() -> dict[str, int]:
    values: dict[str, int] = {}
    value = 10
    forbidden = {11, 22, 33}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        while value in forbidden:
            value += 1
        values[letter] = value
        value += 1
    return values


ISO_LETTER_VALUE = _build_iso_letter_values()


def _normalize_text(text: str | None) -> str:
    return (text or "").strip().upper().replace(" ", "")


def _char_value(char: str) -> int:
    char = char.upper()
    if char.isdigit():
        return int(char)
    return ISO_LETTER_VALUE[char]


def iso6346_check_digit(first_ten: str) -> int:
    if len(first_ten) != 10:
        raise ValueError("ISO 6346 check digit needs exactly 10 first characters.")
    total = sum(_char_value(first_ten[i]) * (2**i) for i in range(10))
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def iso_number_format_valid(number: str) -> bool:
    return bool(ISO_NUMBER_RE.fullmatch(_normalize_text(number)))


def iso_number_check_valid(number: str) -> bool:
    number = _normalize_text(number)
    if not iso_number_format_valid(number):
        return False
    try:
        expected = iso6346_check_digit(number[:10])
    except (KeyError, ValueError):
        return False
    return int(number[10]) == expected


def format_result(primary: str, size_type_code: str | None, elapsed_ms: float) -> str:
    primary = _normalize_text(primary)
    size_type = _normalize_text(size_type_code)
    payload = {
        "result": [
            {"label": "owner_code", "text": primary[:4] if len(primary) >= 4 else ""},
            {"label": "registration_number", "text": primary[4:10] if len(primary) >= 10 else primary[4:]},
            {"label": "check_digit", "text": primary[10] if len(primary) >= 11 else ""},
            {"label": "type_size_code", "text": size_type},
        ],
        "elapsed_ms": round(elapsed_ms, 2),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source.resolve()] if source.suffix.lower() in IMAGE_SUFFIXES else []
    if source.is_dir():
        return sorted(
            p.resolve()
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def _collect_single_image(source: Path) -> Path | None:
    paths = _collect_images(source)
    if len(paths) == 1:
        return paths[0]
    return None


def _camera_pair_in_dir(folder: Path) -> tuple[Path, Path] | None:
    images = _collect_images(folder)
    if len(images) != 2:
        return None
    cam1 = next((p for p in images if "camera1" in p.name.lower()), None)
    cam2 = next((p for p in images if "camera2" in p.name.lower()), None)
    if cam1 is not None and cam2 is not None:
        return cam1, cam2
    return images[0], images[1]


def _pair_sort_key(path: Path) -> tuple[int, int | str]:
    if path.name.isdigit():
        return (0, int(path.name))
    return (1, path.name)


def _collect_pair_jobs(source: Path) -> list[tuple[str, Path, Path]]:
    source = source.resolve()
    direct = _camera_pair_in_dir(source)
    if direct is not None:
        cam1, cam2 = direct
        return [(source.name, cam1, cam2)]
    if not source.is_dir():
        return []

    jobs: list[tuple[str, Path, Path]] = []
    for sub in sorted(source.iterdir(), key=_pair_sort_key):
        if not sub.is_dir():
            continue
        found = _camera_pair_in_dir(sub)
        if found is None:
            continue
        cam1, cam2 = found
        jobs.append((sub.name, cam1, cam2))
    return jobs


def _explicit_pair_job(source1: Path, source2: Path) -> tuple[str, Path, Path] | None:
    cam1 = _collect_single_image(source1)
    cam2 = _collect_single_image(source2)
    if cam1 is None or cam2 is None:
        return None
    return ("pair", cam1, cam2)


def _score_at(read: CameraRead, index: int) -> tuple[str, float] | None:
    if index >= len(read.char_scores):
        return None
    char, conf = read.char_scores[index]
    char = _normalize_text(char)
    if not char:
        return None
    return char, conf


def _choose_position(cam1: CameraRead, cam2: CameraRead, index: int) -> PositionChoice:
    s1 = _score_at(cam1, index)
    s2 = _score_at(cam2, index)

    if s1 is None and s2 is None:
        return PositionChoice(index, "", 0.0, "missing", None, None)
    if s2 is None:
        return PositionChoice(index, s1[0], s1[1], "camera1", s1, None)
    if s1 is None:
        return PositionChoice(index, s2[0], s2[1], "camera2", None, s2)
    if s2[1] > s1[1]:
        return PositionChoice(index, s2[0], s2[1], "camera2", s1, s2)
    return PositionChoice(index, s1[0], s1[1], "camera1", s1, s2)


def merge_by_position_confidence(cam1: CameraRead, cam2: CameraRead) -> tuple[str, list[PositionChoice]]:
    choices = [_choose_position(cam1, cam2, i) for i in range(11)]
    return "".join(choice.char for choice in choices), choices


def _clean_size_type_code(value: str | None) -> str | None:
    raw = _normalize_text(value)
    if not raw:
        return None
    if _is_size_type_code(raw):
        return raw
    # Common case after layout extraction: an extra digit is attached, e.g. 722G1 -> 22G1.
    for size in (4, 3, 2):
        for start in range(0, len(raw) - size + 1):
            candidate = raw[start : start + size]
            if _is_size_type_code(candidate):
                return candidate
    return raw


def _is_size_type_code(value: str) -> bool:
    text = _normalize_text(value)
    if len(text) < 2 or len(text) > 5:
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    if len(text) == 4 and SIZE_TYPE_4_RE.fullmatch(text):
        return True
    if len(text) == 4 and SIZE_TYPE_4_LETTERS_RE.fullmatch(text):
        return True
    if len(text) in (2, 3) and re.fullmatch(r"[0-9A-Z]+", text):
        return any(ch.isalpha() for ch in text)
    return False


def pick_size_type_code(cam1: CameraRead, cam2: CameraRead, winner: str | None) -> str | None:
    codes = {
        "camera1": _clean_size_type_code(cam1.size_type_code),
        "camera2": _clean_size_type_code(cam2.size_type_code),
    }
    if winner in codes and codes[winner]:
        return codes[winner]
    return codes["camera1"] or codes["camera2"]


def merge_two_camera_reads(cam1: CameraRead, cam2: CameraRead) -> ProbabilisticMergeRead:
    text1 = _normalize_text(cam1.primary_number)
    text2 = _normalize_text(cam2.primary_number)

    if text1 and text1 == text2 and iso_number_check_valid(text1):
        return ProbabilisticMergeRead(
            text1,
            True,
            pick_size_type_code(cam1, cam2, "camera1"),
            "both_agree",
            cam1,
            cam2,
            [],
        )

    if cam1.check_ok and not cam2.check_ok:
        return ProbabilisticMergeRead(
            text1,
            True,
            pick_size_type_code(cam1, cam2, "camera1"),
            "camera1_valid",
            cam1,
            cam2,
            [],
        )

    if cam2.check_ok and not cam1.check_ok:
        return ProbabilisticMergeRead(
            text2,
            True,
            pick_size_type_code(cam1, cam2, "camera2"),
            "camera2_valid",
            cam1,
            cam2,
            [],
        )

    merged, choices = merge_by_position_confidence(cam1, cam2)
    merged_ok = iso_number_check_valid(merged)
    winner = None
    cam1_votes = sum(1 for choice in choices if choice.source == "camera1")
    cam2_votes = sum(1 for choice in choices if choice.source == "camera2")
    if cam1_votes > cam2_votes:
        winner = "camera1"
    elif cam2_votes > cam1_votes:
        winner = "camera2"

    return ProbabilisticMergeRead(
        merged if iso_number_format_valid(merged) else "",
        merged_ok,
        pick_size_type_code(cam1, cam2, winner),
        "position_confidence_merge",
        cam1,
        cam2,
        choices,
    )


def predict_pair_with_probability_merge(
    model,
    cam1_path: Path,
    cam2_path: Path,
    *,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: float | None,
) -> ProbabilisticMergeRead:
    cam1 = predict_camera_read(model, cam1_path, conf=conf, iou=iou, max_det=max_det, merge_iou=merge_iou)
    cam2 = predict_camera_read(model, cam2_path, conf=conf, iou=iou, max_det=max_det, merge_iou=merge_iou)
    return merge_two_camera_reads(cam1, cam2)


def _print_verbose(pair_name: str, read: ProbabilisticMergeRead) -> None:
    print(
        f"[{pair_name}] camera1={read.camera1.primary_number!r} "
        f"ok={read.camera1.check_ok} layout={read.camera1.layout} type={read.camera1.size_type_code!r}",
        file=sys.stderr,
    )
    print(
        f"[{pair_name}] camera2={read.camera2.primary_number!r} "
        f"ok={read.camera2.check_ok} layout={read.camera2.layout} type={read.camera2.size_type_code!r}",
        file=sys.stderr,
    )
    print(
        f"[{pair_name}] fusion={read.fusion} result={read.primary_number!r} "
        f"ok={read.check_ok} type={read.size_type_code!r}",
        file=sys.stderr,
    )
    for choice in read.choices:
        c1 = "-" if choice.cam1 is None else f"{choice.cam1[0]}:{choice.cam1[1]:.3f}"
        c2 = "-" if choice.cam2 is None else f"{choice.cam2[0]}:{choice.cam2[1]:.3f}"
        print(
            f"[{pair_name}] pos={choice.index + 1:02d} "
            f"chosen={choice.char!r}:{choice.conf:.3f} from={choice.source} "
            f"cam1={c1} cam2={c2}",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-camera ISO 6346 reader with explicit per-position confidence merge.",
    )
    parser.add_argument("--weights", "-w", type=Path, required=True, help="Path to YOLO .pt weights.")
    parser.add_argument("--source", "-s", type=Path, default=None, help="Directory with one pair or pair folders.")
    parser.add_argument("--source1", "-s1", type=Path, default=None, help="First image.")
    parser.add_argument("--source2", "-s2", type=Path, default=None, help="Second image.")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--merge-iou", type=float, default=DEFAULT_MERGE_IOU)
    parser.add_argument("--verbose", action="store_true", help="Print per-position confidence choices.")
    args = parser.parse_args()

    if args.source2 is not None:
        source1 = args.source1 or args.source
        if source1 is None:
            print("Use -s1 IMAGE1 -s2 IMAGE2, or -s DIR_WITH_PAIRS.", file=sys.stderr)
            return 1
        explicit = _explicit_pair_job(source1, args.source2)
        if explicit is None:
            print(f"Need exactly one image for source1 and source2: {source1}, {args.source2}", file=sys.stderr)
            return 1
        jobs = [explicit]
    elif args.source is not None:
        jobs = _collect_pair_jobs(args.source)
    elif args.source1 is not None:
        jobs = _collect_pair_jobs(args.source1)
    else:
        print("Use -s1 IMAGE1 -s2 IMAGE2, or -s DIR_WITH_PAIRS.", file=sys.stderr)
        return 1

    if not jobs:
        print("No image pairs found.", file=sys.stderr)
        return 1

    merge_iou: float | None = None if args.merge_iou <= 0 else args.merge_iou

    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))

    for pair_name, cam1_path, cam2_path in jobs:
        t0 = time.perf_counter()
        read = predict_pair_with_probability_merge(
            model,
            cam1_path,
            cam2_path,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            merge_iou=merge_iou,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if args.verbose:
            _print_verbose(pair_name, read)
        print(format_result(read.primary_number, read.size_type_code, elapsed_ms))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
