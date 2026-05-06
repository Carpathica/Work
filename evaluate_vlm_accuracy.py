#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request


TARGET_FIELDS = ("owner_code", "registration_number", "check_digit", "type_size_code")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).upper()
    return "".join(ch for ch in text if ch.isalnum())


def normalize_fields(fields: Dict[str, Any]) -> Dict[str, str]:
    return {name: normalize_text(fields.get(name, "")) for name in TARGET_FIELDS}


def parse_fields(value: Any) -> Dict[str, str]:
    if isinstance(value, dict):
        return normalize_fields(value)
    if isinstance(value, list):
        mapped: Dict[str, Any] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if label in TARGET_FIELDS:
                mapped[label] = item.get("text", "")
        return normalize_fields(mapped)
    return normalize_fields({})


def iso_from_fields(fields: Dict[str, str]) -> str:
    return (
        f"{fields.get('owner_code', '')}"
        f"{fields.get('registration_number', '')}"
        f"{fields.get('check_digit', '')}"
    ).strip()


def load_cases(dataset_path: Path, max_samples: int, default_scenario: str) -> List[Dict[str, Any]]:
    dataset_dir = dataset_path.parent.resolve()
    rows = dataset_path.read_text(encoding="utf-8").splitlines()
    cases: List[Dict[str, Any]] = []

    for index, line in enumerate(rows, start=1):
        if not line.strip():
            continue
        raw = json.loads(line)

        case_id = str(raw.get("id") or f"case_{index}")
        scenario = str(raw.get("scenario") or default_scenario)
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        metadata.setdefault("scenario", scenario)

        image_refs = raw.get("images", raw.get("image"))
        if image_refs is None:
            continue
        if isinstance(image_refs, str):
            image_refs = [image_refs]
        if not isinstance(image_refs, list):
            continue

        image_paths: List[Path] = []
        for ref in image_refs:
            if not isinstance(ref, str):
                continue
            p = Path(ref)
            if not p.is_absolute():
                p = (dataset_dir / p).resolve()
            else:
                p = p.resolve()
            image_paths.append(p)

        expected_raw = raw.get("expected")
        expected = parse_fields(expected_raw) if expected_raw is not None else parse_fields({})
        has_expected = expected_raw is not None

        cases.append(
            {
                "case_id": case_id,
                "metadata": metadata,
                "image_paths": image_paths,
                "expected": expected,
                "has_expected": has_expected,
            }
        )

        if max_samples > 0 and len(cases) >= max_samples:
            break

    return cases


def http_json_post(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    req = request.Request(url=url, data=body, headers=req_headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            content = resp.read().decode("utf-8", errors="replace")
            data = json.loads(content) if content else {}
            return status, data
    except error.HTTPError as exc:
        content = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(content) if content else {}
        except Exception:
            data = {"raw": content}
        return int(exc.code), data


def build_multipart_body(
    metadata_json: str,
    images: List[Path],
) -> Tuple[bytes, str]:
    boundary = f"----ReleaseDemoBoundary{uuid.uuid4().hex}"
    chunks: List[bytes] = []

    def add_line(value: str) -> None:
        chunks.append(value.encode("utf-8"))

    add_line(f"--{boundary}\r\n")
    add_line('Content-Disposition: form-data; name="metadata"\r\n')
    add_line("Content-Type: text/plain; charset=utf-8\r\n\r\n")
    add_line(metadata_json)
    add_line("\r\n")

    for image_path in images:
        filename = image_path.name
        mime_type, _ = mimetypes.guess_type(str(image_path))
        content_type = mime_type or "application/octet-stream"
        data = image_path.read_bytes()

        add_line(f"--{boundary}\r\n")
        add_line(
            f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
        )
        add_line(f"Content-Type: {content_type}\r\n\r\n")
        chunks.append(data)
        add_line("\r\n")

    add_line(f"--{boundary}--\r\n")
    body = b"".join(chunks)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def http_multipart_post(
    url: str,
    metadata_json: str,
    images: List[Path],
    timeout: float,
    bearer_token: str,
) -> Tuple[int, Dict[str, Any]]:
    body, content_type = build_multipart_body(metadata_json, images)
    headers = {
        "Content-Type": content_type,
        "Authorization": f"Bearer {bearer_token}",
    }
    req = request.Request(url=url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            content = resp.read().decode("utf-8", errors="replace")
            data = json.loads(content) if content else {}
            return status, data
    except error.HTTPError as exc:
        content = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(content) if content else {}
        except Exception:
            data = {"raw": content}
        return int(exc.code), data


def login(base_url: str, username: str, password: str, timeout: float) -> str:
    status, data = http_json_post(
        url=f"{base_url.rstrip('/')}/auth/login",
        payload={"username": username, "password": password},
        timeout=timeout,
    )
    if status != 200:
        raise RuntimeError(f"Login failed: HTTP {status}, body={data}")
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Login succeeded but no access_token in response.")
    return str(token)


def extract_predicted_fields(response_payload: Dict[str, Any]) -> Dict[str, str]:
    content: Any = response_payload.get("result", response_payload)
    if isinstance(content, dict) and "result" in content:
        content = content["result"]
    return parse_fields(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate release-demo VLM accuracy without external Python deps."
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("RELEASE_DEMO_BASE_URL") or "http://localhost:8899",
        help="release-demo base URL.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("RELEASE_DEMO_USERNAME") or "admin@example.com",
        help="release-demo username.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("RELEASE_DEMO_PASSWORD") or "admin123@pass!",
        help="release-demo password.",
    )
    parser.add_argument(
        "--default-scenario",
        default="extract_number_general",
        help="Fallback scenario if row does not define scenario.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional sample limit.")
    parser.add_argument("--debug-rows", action="store_true", help="Include per-case rows in output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cases = load_cases(
        dataset_path=dataset_path,
        max_samples=max(int(args.max_samples), 0),
        default_scenario=args.default_scenario,
    )
    if not cases:
        raise RuntimeError("No valid cases in dataset.")

    token = login(
        base_url=args.base_url,
        username=args.username,
        password=args.password,
        timeout=float(args.timeout),
    )

    started = time.time()
    rows: List[Dict[str, Any]] = []

    recognize_url = f"{args.base_url.rstrip('/')}/api/v1/recognize-multipart"

    for idx, case in enumerate(cases, start=1):
        sample_started = time.perf_counter()
        status_code = 0
        error_name: Optional[str] = None
        predicted = parse_fields({})
        request_id = None

        try:
            image_paths: List[Path] = case["image_paths"]
            image_paths = [p for p in image_paths if p.exists()]
            if not image_paths:
                raise FileNotFoundError("No existing images for case.")

            metadata_json = json.dumps(case["metadata"], ensure_ascii=False)
            status_code, payload = http_multipart_post(
                url=recognize_url,
                metadata_json=metadata_json,
                images=image_paths,
                timeout=float(args.timeout),
                bearer_token=token,
            )

            if status_code == 401:
                token = login(
                    base_url=args.base_url,
                    username=args.username,
                    password=args.password,
                    timeout=float(args.timeout),
                )
                status_code, payload = http_multipart_post(
                    url=recognize_url,
                    metadata_json=metadata_json,
                    images=image_paths,
                    timeout=float(args.timeout),
                    bearer_token=token,
                )

            if status_code == 200:
                request_id = payload.get("request_id")
                predicted = extract_predicted_fields(payload)
            else:
                error_name = f"http_{status_code}"
        except Exception as exc:  # noqa: BLE001
            error_name = type(exc).__name__

        latency_ms = (time.perf_counter() - sample_started) * 1000.0

        expected: Dict[str, str] = case["expected"]
        expected_iso = iso_from_fields(expected)
        predicted_iso = iso_from_fields(predicted)

        has_expected = bool(case["has_expected"])
        full_match = (
            has_expected and all(predicted.get(field, "") == expected.get(field, "") for field in TARGET_FIELDS)
        )
        iso_match = has_expected and predicted_iso.upper() == expected_iso.upper()

        rows.append(
            {
                "case_id": case["case_id"],
                "status_code": status_code,
                "request_id": request_id,
                "error": error_name,
                "latency_ms": latency_ms,
                "expected": expected,
                "predicted": predicted,
                "full_match": full_match if has_expected else None,
                "iso_match": iso_match if has_expected else None,
            }
        )
        print(f"[{idx}/{len(cases)}] case={case['case_id']} status={status_code} error={error_name or 'none'}")

    finished = time.time()

    scored = [r for r in rows if r.get("full_match") is not None]
    scored_iso = [r for r in rows if r.get("iso_match") is not None]
    full_correct = sum(1 for r in scored if r["full_match"])
    iso_correct = sum(1 for r in scored_iso if r["iso_match"])

    field_accuracy: Dict[str, Optional[float]] = {}
    for field in TARGET_FIELDS:
        if not scored:
            field_accuracy[field] = None
        else:
            hits = sum(1 for r in scored if r["predicted"].get(field) == r["expected"].get(field))
            field_accuracy[field] = hits / len(scored)

    latencies = [float(r["latency_ms"]) for r in rows]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    status_codes = Counter(int(r["status_code"]) for r in rows)

    report = {
        "dataset": str(dataset_path),
        "samples_total": len(rows),
        "samples_scored": len(scored),
        "full_match_accuracy": (full_correct / len(scored)) if scored else 0.0,
        "iso_accuracy": (iso_correct / len(scored_iso)) if scored_iso else 0.0,
        "field_accuracy": field_accuracy,
        "status_codes": {str(k): v for k, v in status_codes.items()},
        "avg_latency_ms": avg_latency,
        "started_at_unix": started,
        "finished_at_unix": finished,
        "duration_sec": finished - started,
        "rows": rows if args.debug_rows else [],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

