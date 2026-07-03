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
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    parser.add_argument(
        "--date-from",
        default=None,
        help="Start date filter, inclusive, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="End date filter, inclusive, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--datetime-from",
        default=None,
        help="Start datetime filter, inclusive, e.g. 2026-06-22T00:00:00.",
    )
    parser.add_argument(
        "--datetime-to",
        default=None,
        help="End datetime filter, inclusive, e.g. 2026-06-24T23:59:59.",
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


def parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def parse_datetime_arg(value: str, name: str) -> datetime:
    parsed = parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"{name} must be an ISO datetime, got {value!r}")
    return parsed


def parse_date_start(value: str, name: str) -> datetime:
    try:
        return datetime.fromisoformat(value).replace(hour=0, minute=0, second=0, microsecond=0)
    except ValueError as exc:
        raise ValueError(f"{name} must be a date in YYYY-MM-DD format, got {value!r}") from exc


def build_time_filter(args: argparse.Namespace) -> tuple[datetime | None, datetime | None, dict[str, Any]]:
    start: datetime | None = None
    end: datetime | None = None
    source = "none"

    if args.datetime_from:
        start = parse_datetime_arg(args.datetime_from, "--datetime-from")
        source = "datetime"
    elif args.date_from:
        start = parse_date_start(args.date_from, "--date-from")
        source = "date"

    if args.datetime_to:
        end = parse_datetime_arg(args.datetime_to, "--datetime-to")
        source = "datetime"
    elif args.date_to:
        end = parse_date_start(args.date_to, "--date-to") + timedelta(days=1) - timedelta(microseconds=1)
        source = "date" if source == "none" else source

    if start and end and start > end:
        raise ValueError("Datetime filter start must be earlier than or equal to end.")

    return start, end, {
        "active": bool(start or end),
        "source": source,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "datetime_from": args.datetime_from,
        "datetime_to": args.datetime_to,
        "effective_from": start.isoformat(timespec="seconds") if start else None,
        "effective_to": end.isoformat(timespec="seconds") if end else None,
    }


def filter_records_by_time(
    records: Sequence[dict[str, Any]],
    start: datetime | None,
    end: datetime | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if start is None and end is None:
        return list(records), {
            "kept": len(records),
            "dropped_missing_or_invalid_timestamp": 0,
            "dropped_before_start": 0,
            "dropped_after_end": 0,
        }

    out: list[dict[str, Any]] = []
    stats = {
        "kept": 0,
        "dropped_missing_or_invalid_timestamp": 0,
        "dropped_before_start": 0,
        "dropped_after_end": 0,
    }
    for record in records:
        timestamp = parse_timestamp(record.get("timestamp"))
        if timestamp is None:
            stats["dropped_missing_or_invalid_timestamp"] += 1
            continue
        if start is not None and timestamp < start:
            stats["dropped_before_start"] += 1
            continue
        if end is not None and timestamp > end:
            stats["dropped_after_end"] += 1
            continue
        stats["kept"] += 1
        out.append(record)
    return out, stats


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


def summarize_camera_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    bucket_summary = summarize_buckets(row["bucket"] for row in rows)
    total = len(rows)
    number_ok = sum(1 for row in rows if row["check_ok"])
    type_present = sum(1 for row in rows if row["type_present"])
    type_format_ok = sum(1 for row in rows if row["type_format_ok"])
    type_allowlist_ok = sum(1 for row in rows if row["type_allowlist_ok"])
    full_ok = sum(1 for row in rows if row["bucket"] == "full_number_and_type_ok")
    attempts = sum(1 for row in rows if row["primary_number"] or row["detections_count"] > 0)

    return {
        **bucket_summary,
        "attempts": attempts,
        "number_ok": number_ok,
        "number_error": bucket_summary["counts"]["number_error"],
        "type_present": type_present,
        "type_missing": total - type_present,
        "type_format_ok": type_format_ok,
        "type_format_error": total - type_format_ok,
        "type_allowlist_ok": type_allowlist_ok,
        "type_allowlist_error": total - type_allowlist_ok,
        "full_number_and_type_ok": full_ok,
        "recognition_rates": {
            "number_ok_rate": percent(number_ok, total),
            "type_format_ok_rate": percent(type_format_ok, total),
            "type_allowlist_ok_rate": percent(type_allowlist_ok, total),
            "full_number_and_type_ok_rate": percent(full_ok, total),
        },
    }


def summarize_camera_rows_by_camera(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        camera_name = str(row.get("camera") or "unknown")
        grouped[camera_name].append(row)
    return {
        camera_name: summarize_camera_rows(camera_rows_for_name)
        for camera_name, camera_rows_for_name in sorted(grouped.items())
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
    camera_type_present = sum(1 for row in camera_level_rows if row["type_present"])
    camera_type_missing = len(camera_level_rows) - camera_type_present
    type_values_total_occurrences = sum(int(row["count"]) for row in type_values)

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
        "camera_metrics_by_camera": summarize_camera_rows_by_camera(camera_level_rows),
        "type_size_code_counts": {
            "camera_rows_total": len(camera_level_rows),
            "camera_rows_with_type": camera_type_present,
            "camera_rows_without_type": camera_type_missing,
            "unique_codes_seen": len(type_values),
            "total_code_occurrences": type_values_total_occurrences,
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
    decision_filtered_records = filter_records(all_records, args.include_decisions)
    time_start, time_end, time_filter = build_time_filter(args)
    records, time_filter_stats = filter_records_by_time(
        decision_filtered_records,
        time_start,
        time_end,
    )
    cams = camera_rows(records, valid_codes)
    pairs = pair_rows(records, valid_codes)
    type_values = type_value_rows(cams, valid_codes)
    summary = build_summary(records, pairs, cams, type_values, valid_codes)
    summary["records_input_total"] = len(all_records)
    summary["records_after_decision_filter"] = len(decision_filtered_records)
    summary["records_after_time_filter"] = len(records)
    summary["include_decisions"] = args.include_decisions or []
    summary["time_filter"] = time_filter
    summary["time_filter_stats"] = time_filter_stats

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
