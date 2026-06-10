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
BACKENDS = ("vlm", "yolo", "yolo-scenario")


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
    rows = dataset_path.read_text(encoding="utf-8-sig").splitlines()
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

        cases.append(
            {
                "case_id": case_id,
                "metadata": metadata,
                "image_paths": image_paths,
                "expected": expected,
                "has_expected": expected_raw is not None,
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
        return parse_http_error(exc)


def parse_http_error(exc: error.HTTPError) -> Tuple[int, Dict[str, Any]]:
    content = exc.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(content) if content else {}
    except Exception:
        data = {"raw": content}
    return int(exc.code), data


def form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def build_multipart_body(
    fields: Dict[str, Any],
    images: List[Path],
) -> Tuple[bytes, str]:
    boundary = f"----ReleaseDemoBoundary{uuid.uuid4().hex}"
    chunks: List[bytes] = []

    def add_text(value: str) -> None:
        chunks.append(value.encode("utf-8"))

    for name, value in fields.items():
        add_text(f"--{boundary}\r\n")
        add_text(f'Content-Disposition: form-data; name="{name}"\r\n')
        add_text("Content-Type: text/plain; charset=utf-8\r\n\r\n")
        add_text(form_value(value))
        add_text("\r\n")

    for image_path in images:
        filename = image_path.name
        mime_type, _ = mimetypes.guess_type(str(image_path))
        content_type = mime_type or "application/octet-stream"
        data = image_path.read_bytes()

        add_text(f"--{boundary}\r\n")
        add_text(f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n')
        add_text(f"Content-Type: {content_type}\r\n\r\n")
        chunks.append(data)
        add_text("\r\n")

    add_text(f"--{boundary}--\r\n")
    body = b"".join(chunks)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def http_multipart_post(
    url: str,
    fields: Dict[str, Any],
    images: List[Path],
    timeout: float,
    bearer_token: str,
) -> Tuple[int, Dict[str, Any]]:
    body, content_type = build_multipart_body(fields=fields, images=images)
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
        return parse_http_error(exc)


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

    # release-demo YOLO endpoints return {"result": {"result": [...]}}.
    if isinstance(content, dict) and "result" in content:
        content = content["result"]

    return parse_fields(content)


def endpoint_for_backend(base_url: str, backend: str) -> str:
    root = base_url.rstrip("/")
    if backend == "vlm":
        return f"{root}/api/v1/recognize-multipart"
    if backend == "yolo":
        return f"{root}/api/v1/yolo-recognize-multipart"
    if backend == "yolo-scenario":
        return f"{root}/api/v1/yolo-recognize-by-scenario"
    raise ValueError(f"Unsupported backend: {backend}")


def request_fields_for_case(args: argparse.Namespace, case: Dict[str, Any]) -> Dict[str, Any]:
    metadata = case["metadata"]

    if args.backend == "vlm":
        return {"metadata": json.dumps(metadata, ensure_ascii=False)}

    if args.backend == "yolo":
        if not args.yolo_model:
            raise ValueError("--yolo-model is required when --backend yolo")
        return {
            "script": args.yolo_script,
            "model": args.yolo_model,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "merge_iou": args.merge_iou,
            "agnostic_nms": args.agnostic_nms,
        }

    if args.backend == "yolo-scenario":
        scenario = args.yolo_scenario or str(metadata.get("scenario") or "")
        if not scenario:
            raise ValueError(
                "--yolo-scenario is required when case metadata has no scenario"
            )
        return {"scenario": scenario}

    raise ValueError(f"Unsupported backend: {args.backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate release-demo VLM or YOLO recognition accuracy."
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
        "--backend",
        choices=BACKENDS,
        default="vlm",
        help="Recognition backend to evaluate.",
    )
    parser.add_argument(
        "--default-scenario",
        default="extract_number_general",
        help="Fallback scenario if row does not define scenario.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout seconds.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional sample limit.",
    )
    parser.add_argument(
        "--debug-rows",
        action="store_true",
        help="Include per-case rows in JSON output.",
    )
    parser.add_argument(
        "--print-mismatches",
        action="store_true",
        help="Print expected vs predicted values for failed scored cases.",
    )

    yolo = parser.add_argument_group("YOLO backend options")
    yolo.add_argument(
        "--yolo-script",
        default="read_container",
        choices=("read_container", "read_container_KP"),
        help="YOLO script for --backend yolo.",
    )
    yolo.add_argument(
        "--yolo-model",
        default=os.getenv("RELEASE_DEMO_YOLO_MODEL"),
        help="YOLO model filename/path for --backend yolo.",
    )
    yolo.add_argument(
        "--yolo-scenario",
        default=os.getenv("RELEASE_DEMO_YOLO_SCENARIO"),
        help="Active YOLO scenario name for --backend yolo-scenario.",
    )
    yolo.add_argument("--conf", type=float, default=0.15, help="YOLO confidence.")
    yolo.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU.")
    yolo.add_argument("--max-det", type=int, default=300, help="YOLO max detections.")
    yolo.add_argument(
        "--merge-iou",
        type=float,
        default=0.35,
        help="Postprocessing IoU merge threshold. Use 0 to disable.",
    )
    yolo.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic YOLO NMS.",
    )

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
    recognize_url = endpoint_for_backend(args.base_url, args.backend)

    for idx, case in enumerate(cases, start=1):
        sample_started = time.perf_counter()
        status_code = 0
        error_name: Optional[str] = None
        error_detail: Optional[str] = None
        predicted = parse_fields({})
        request_id = None

        try:
            image_paths: List[Path] = case["image_paths"]
            image_paths = [p for p in image_paths if p.exists()]
            if not image_paths:
                raise FileNotFoundError("No existing images for case.")

            fields = request_fields_for_case(args, case)
            status_code, payload = http_multipart_post(
                url=recognize_url,
                fields=fields,
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
                    fields=fields,
                    images=image_paths,
                    timeout=float(args.timeout),
                    bearer_token=token,
                )

            if status_code == 200:
                request_id = payload.get("request_id")
                predicted = extract_predicted_fields(payload)
            else:
                error_name = f"http_{status_code}"
                error_detail = json.dumps(payload, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            error_name = type(exc).__name__
            error_detail = str(exc)

        latency_ms = (time.perf_counter() - sample_started) * 1000.0

        expected: Dict[str, str] = case["expected"]
        expected_iso = iso_from_fields(expected)
        predicted_iso = iso_from_fields(predicted)

        has_expected = bool(case["has_expected"])
        full_match = has_expected and all(
            predicted.get(field, "") == expected.get(field, "")
            for field in TARGET_FIELDS
        )
        iso_match = has_expected and predicted_iso.upper() == expected_iso.upper()

        row = {
            "case_id": case["case_id"],
            "backend": args.backend,
            "status_code": status_code,
            "request_id": request_id,
            "error": error_name,
            "error_detail": error_detail,
            "latency_ms": latency_ms,
            "expected": expected,
            "predicted": predicted,
            "full_match": full_match if has_expected else None,
            "iso_match": iso_match if has_expected else None,
        }
        rows.append(row)

        print(
            f"[{idx}/{len(cases)}] case={case['case_id']} "
            f"backend={args.backend} status={status_code} "
            f"error={error_name or 'none'} iso_match={row['iso_match']}"
        )

        if args.print_mismatches and has_expected and not full_match:
            print(
                f"[BAD] {case['case_id']} | "
                f"exp={expected} | pred={predicted} | "
                f"iso_exp={expected_iso} iso_pred={predicted_iso}"
            )

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
            hits = sum(
                1
                for r in scored
                if r["predicted"].get(field) == r["expected"].get(field)
            )
            field_accuracy[field] = hits / len(scored)

    latencies = [float(r["latency_ms"]) for r in rows]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    status_codes = Counter(int(r["status_code"]) for r in rows)

    report = {
        "dataset": str(dataset_path),
        "backend": args.backend,
        "endpoint": recognize_url,
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
