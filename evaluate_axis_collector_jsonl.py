#!/usr/bin/env python3
"""
Evaluate axis_dual_container_collector JSONL logs.

This evaluator does not use ground truth for container numbers. A number is
treated as correct when the collector/read_container result has check_ok=True.
Type/size codes are evaluated in two ways:
  - format validity: 2 digits + alnum + digit (e.g. 22G1)
  - allowlist validity: code is present in a user-editable list
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


TYPE_SIZE_RE = re.compile(r"^[0-9]{2}[A-Z0-9][0-9]$")


@dataclass(frozen=True)
class TypeStatus:
    code: str
    present: bool
    format_ok: bool
    allowlist_ok: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate axis_dual_container_collector JSONL metrics."
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        required=True,
        type=Path,
        help="Collector JSONL file(s).",
    )
    parser.add_argument(
        "--valid-type-codes",
        type=Path,
        default=Path("configs/valid_type_size_codes.txt"),
        help="Allowlist of valid type_size_code values, one per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: runs/axis_metrics_<timestamp>.",
    )
    parser.add_argument(
        "--include-decisions",
        nargs="+",
        default=None,
        help="Optional decision filter, e.g. --include-decisions save skip.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                row = json.loads(text)
                if not isinstance(row, dict):
                    continue
                row["_source_jsonl"] = str(path)
                row["_source_line"] = line_no
                rows.append(row)
    return rows


def filter_records(
    records: Sequence[dict[str, Any]],
    include_decisions: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if not include_decisions:
        return list(records)
    allowed = {str(item).strip().lower() for item in include_decisions if str(item).strip()}
    return [
        record
        for record in records
        if str(record.get("decision") or "").strip().lower() in allowed
    ]


def load_valid_type_codes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        code = normalize_text(line.split("#", 1)[0])
        if code:
            out.add(code)
    return out


def type_status(value: Any, valid_codes: set[str]) -> TypeStatus:
    code = normalize_text(value)
    present = bool(code)
    format_ok = bool(TYPE_SIZE_RE.fullmatch(code)) if present else False
    allowlist_ok = code in valid_codes if present else False
    return TypeStatus(code, present, format_ok, allowlist_ok)


def camera_has_attempt(camera: dict[str, Any]) -> bool:
    return bool(normalize_text(camera.get("primary_number"))) or int(camera.get("detections_count") or 0) > 0


def camera_bucket(camera: dict[str, Any], valid_codes: set[str]) -> str:
    check_ok = bool(camera.get("check_ok"))
    status = type_status(camera.get("size_type_code"), valid_codes)
    if check_ok and status.allowlist_ok:
        return "full_number_and_type_ok"
    if check_ok:
        return "type_size_code_error"
    if camera_has_attempt(camera):
        return "number_error"
    return "no_recognition"


def choose_pair_type(cameras: Sequence[dict[str, Any]], valid_codes: set[str]) -> tuple[TypeStatus, str]:
    check_ok_cameras = [c for c in cameras if bool(c.get("check_ok"))]
    candidates = check_ok_cameras or list(cameras)

    with_type = [c for c in candidates if normalize_text(c.get("size_type_code"))]
    source_kind = "check_ok_camera" if check_ok_cameras else "any_camera"
    if not with_type and check_ok_cameras:
        with_type = [c for c in cameras if normalize_text(c.get("size_type_code"))]
        source_kind = "non_check_ok_camera"

    if not with_type:
        return type_status("", valid_codes), "none"

    statuses = [(c, type_status(c.get("size_type_code"), valid_codes)) for c in with_type]
    statuses.sort(
        key=lambda item: (
            not item[1].allowlist_ok,
            not item[1].format_ok,
            -int(item[0].get("detections_count") or 0),
            str(item[0].get("camera") or ""),
        )
    )
    chosen_camera, chosen_status = statuses[0]
    return chosen_status, f"{source_kind}:{chosen_camera.get('camera', '')}"


def pair_bucket(cameras: Sequence[dict[str, Any]], valid_codes: set[str]) -> str:
    number_ok = any(bool(c.get("check_ok")) for c in cameras)
    status, _source = choose_pair_type(cameras, valid_codes)
    if number_ok and status.allowlist_ok:
        return "full_number_and_type_ok"
    if number_ok:
        return "type_size_code_error"
    if any(camera_has_attempt(c) for c in cameras):
        return "number_error"
    return "no_recognition"


def percent(part: int, total: int) -> float:
    return (part / total) if total else 0.0


def summarize_buckets(buckets: Iterable[str]) -> dict[str, Any]:
    counts = Counter(buckets)
    total = sum(counts.values())
    ordered = {
        "full_number_and_type_ok": counts.get("full_number_and_type_ok", 0),
        "number_error": counts.get("number_error", 0),
        "type_size_code_error": counts.get("type_size_code_error", 0),
        "no_recognition": counts.get("no_recognition", 0),
    }
    return {
        "total": total,
        "counts": ordered,
        "rates": {key: percent(value, total) for key, value in ordered.items()},
    }


def camera_rows(records: Sequence[dict[str, Any]], valid_codes: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records, start=1):
        cameras = record.get("cameras") if isinstance(record.get("cameras"), list) else []
        for cam in cameras:
            if not isinstance(cam, dict):
                continue
            status = type_status(cam.get("size_type_code"), valid_codes)
            bucket = camera_bucket(cam, valid_codes)
            rows.append(
                {
                    "record_index": record_idx,
                    "source_jsonl": record.get("_source_jsonl", ""),
                    "source_line": record.get("_source_line", ""),
                    "timestamp": record.get("timestamp", ""),
                    "decision": record.get("decision", ""),
                    "reason": record.get("reason", ""),
                    "category": record.get("category", ""),
                    "camera": cam.get("camera", ""),
                    "image": cam.get("image", ""),
                    "primary_number": normalize_text(cam.get("primary_number")),
                    "check_ok": bool(cam.get("check_ok")),
                    "detections_count": int(cam.get("detections_count") or 0),
                    "size_type_code": status.code,
                    "type_present": status.present,
                    "type_format_ok": status.format_ok,
                    "type_allowlist_ok": status.allowlist_ok,
                    "bucket": bucket,
                }
            )
    return rows


def pair_rows(records: Sequence[dict[str, Any]], valid_codes: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records, start=1):
        cameras = [
            cam for cam in (record.get("cameras") or [])
            if isinstance(cam, dict)
        ]
        status, type_source = choose_pair_type(cameras, valid_codes)
        number_ok = any(bool(c.get("check_ok")) for c in cameras)
        attempted = any(camera_has_attempt(c) for c in cameras)
        bucket = pair_bucket(cameras, valid_codes)
        rows.append(
            {
                "record_index": record_idx,
                "source_jsonl": record.get("_source_jsonl", ""),
                "source_line": record.get("_source_line", ""),
                "timestamp": record.get("timestamp", ""),
                "mode": record.get("mode", ""),
                "decision": record.get("decision", ""),
                "reason": record.get("reason", ""),
                "category": record.get("category", ""),
                "number_ok": number_ok,
                "attempted": attempted,
                "chosen_type_size_code": status.code,
                "type_source": type_source,
                "type_present": status.present,
                "type_format_ok": status.format_ok,
                "type_allowlist_ok": status.allowlist_ok,
                "bucket": bucket,
                "camera_count": len(cameras),
                "camera_numbers": "|".join(normalize_text(c.get("primary_number")) for c in cameras),
                "camera_types": "|".join(normalize_text(c.get("size_type_code")) for c in cameras),
                "camera_check_ok": "|".join(str(bool(c.get("check_ok"))) for c in cameras),
                "camera_detections": "|".join(str(int(c.get("detections_count") or 0)) for c in cameras),
            }
        )
    return rows


def type_value_rows(camera_level_rows: Sequence[dict[str, Any]], valid_codes: set[str]) -> list[dict[str, Any]]:
    counts = Counter(row["size_type_code"] for row in camera_level_rows if row["size_type_code"])
    rows = []
    for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        status = type_status(code, valid_codes)
        rows.append(
            {
                "size_type_code": code,
                "count": count,
                "type_format_ok": status.format_ok,
                "type_allowlist_ok": status.allowlist_ok,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    records: Sequence[dict[str, Any]],
    pair_level_rows: Sequence[dict[str, Any]],
    camera_level_rows: Sequence[dict[str, Any]],
    type_values: Sequence[dict[str, Any]],
    valid_type_codes: set[str],
) -> dict[str, Any]:
    decisions = Counter(str(row.get("decision") or "") for row in records)
    categories = Counter(str(row.get("category") or "") for row in records)
    pair_type_format_ok = sum(1 for row in pair_level_rows if row["type_format_ok"])
    pair_type_allowlist_ok = sum(1 for row in pair_level_rows if row["type_allowlist_ok"])
    camera_type_format_ok = sum(1 for row in camera_level_rows if row["type_format_ok"])
    camera_type_allowlist_ok = sum(1 for row in camera_level_rows if row["type_allowlist_ok"])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records_total": len(records),
        "valid_type_codes_count": len(valid_type_codes),
        "decisions": dict(decisions),
        "categories": dict(categories),
        "pair_metrics": {
            **summarize_buckets(row["bucket"] for row in pair_level_rows),
            "type_format_ok": pair_type_format_ok,
            "type_format_error": len(pair_level_rows) - pair_type_format_ok,
            "type_allowlist_ok": pair_type_allowlist_ok,
            "type_allowlist_error": len(pair_level_rows) - pair_type_allowlist_ok,
        },
        "camera_metrics": {
            **summarize_buckets(row["bucket"] for row in camera_level_rows),
            "type_format_ok": camera_type_format_ok,
            "type_format_error": len(camera_level_rows) - camera_type_format_ok,
            "type_allowlist_ok": camera_type_allowlist_ok,
            "type_allowlist_error": len(camera_level_rows) - camera_type_allowlist_ok,
        },
        "type_size_codes_seen": list(type_values),
    }


def default_output_dir() -> Path:
    return Path("runs") / f"axis_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_codes = load_valid_type_codes(args.valid_type_codes)
    all_records = read_jsonl(args.jsonl)
    records = filter_records(all_records, args.include_decisions)
    cams = camera_rows(records, valid_codes)
    pairs = pair_rows(records, valid_codes)
    type_values = type_value_rows(cams, valid_codes)
    summary = build_summary(records, pairs, cams, type_values, valid_codes)
    summary["records_input_total"] = len(all_records)
    summary["include_decisions"] = args.include_decisions or []

    write_csv(output_dir / "rows.csv", cams)
    write_csv(output_dir / "pairs.csv", pairs)
    write_csv(output_dir / "type_size_code_values.csv", type_values)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Reports written to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
