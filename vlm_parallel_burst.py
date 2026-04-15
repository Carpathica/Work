#!/usr/bin/env python3
import argparse
import base64
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single burst test: send 5-6 image recognize requests in parallel."
    )
    parser.add_argument("--base-url", default="http://localhost:8899", help="Web API base URL.")
    parser.add_argument("--username", default="admin@example.com", help="Login username.")
    parser.add_argument("--password", default="admin123@pass!", help="Login password.")
    parser.add_argument(
        "--scenario",
        default="extract_number_container_site",
        help="metadata.scenario value.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image files and/or directories with jpg/jpeg/png.",
    )
    parser.add_argument(
        "--parallel-count",
        type=int,
        default=6,
        help="How many simultaneous requests to send (usually 5 or 6).",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=240.0,
        help="Timeout for one recognize request.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional JSON report path.",
    )
    return parser.parse_args()


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def json_request(
    method: str,
    url: str,
    payload: Optional[dict] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Tuple[int, dict]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = Request(url=url, method=method, data=data, headers=req_headers)

    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            body_bytes = response.read()
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            return status, body
    except HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {"detail": str(e)}
        return int(e.code), body
    except URLError as e:
        return 0, {"detail": f"URLError: {e}"}
    except Exception as e:
        return 0, {"detail": f"Error: {e}"}


def login(base_url: str, username: str, password: str) -> str:
    status, body = json_request(
        method="POST",
        url=f"{base_url}/auth/login",
        payload={"username": username, "password": password},
        timeout=30.0,
    )
    if status != 200:
        raise RuntimeError(f"Login failed: status={status}, body={body}")

    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"Login succeeded but access_token missing: {body}")
    return token


def collect_images(inputs: List[str]) -> List[Path]:
    allowed = {".jpg", ".jpeg", ".png"}
    images: List[Path] = []

    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix.lower() in allowed:
            images.append(p.resolve())
            continue

        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in allowed:
                    images.append(f.resolve())

    deduped: List[Path] = []
    seen = set()
    for img in images:
        key = str(img).lower()
        if key not in seen:
            deduped.append(img)
            seen.add(key)
    return deduped


def image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def recognize_one(
    base_url: str,
    token: str,
    scenario: str,
    image_path: Path,
    timeout_sec: float,
) -> dict:
    started = time.perf_counter()
    payload = {
        "images": [image_to_base64(image_path)],
        "metadata": {"scenario": scenario},
    }
    status, body = json_request(
        method="POST",
        url=f"{base_url}/api/v1/recognize",
        payload=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout_sec,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    return {
        "image": str(image_path),
        "status_code": status,
        "ok": 200 <= status < 300,
        "duration_ms": round(duration_ms, 2),
        "request_id": body.get("request_id"),
        "error": body.get("detail") if not (200 <= status < 300) else None,
    }


def read_health(base_url: str) -> dict:
    status, body = json_request("GET", f"{base_url}/health", timeout=15.0)
    return {
        "status_code": status,
        "ok": 200 <= status < 300,
        "body": body,
    }


def extract_health_brief(health: dict) -> dict:
    if not health.get("ok"):
        return {"ok": False, "status_code": health.get("status_code")}

    body = health.get("body", {})
    resources = body.get("resources", {})
    http_data = body.get("http", {})
    return {
        "ok": True,
        "status_code": health.get("status_code"),
        "cpu_percent": resources.get("cpu_percent"),
        "rss_mb": resources.get("rss_mb"),
        "requests_total": http_data.get("requests_total"),
        "errors_total": http_data.get("errors_total"),
        "avg_request_time_ms": http_data.get("avg_request_time_ms"),
    }


def main() -> int:
    args = parse_args()
    base_url = normalize_base_url(args.base_url)

    if args.parallel_count < 1:
        raise ValueError("--parallel-count must be >= 1")

    all_images = collect_images(args.images)
    if len(all_images) < args.parallel_count:
        raise RuntimeError(
            f"Need at least {args.parallel_count} images, found only {len(all_images)}."
        )

    selected = all_images[: args.parallel_count]
    print(f"[info] Using {len(selected)} images for simultaneous burst:")
    for i, img in enumerate(selected, start=1):
        print(f"  {i}. {img}")

    token = login(base_url, args.username, args.password)
    print("[info] Login successful.")

    health_before = read_health(base_url)
    burst_started_utc = datetime.utcnow().isoformat() + "Z"
    burst_started = time.perf_counter()
    results: List[dict] = []

    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = [
            executor.submit(
                recognize_one,
                base_url,
                token,
                args.scenario,
                img,
                args.request_timeout_sec,
            )
            for img in selected
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                "[result] "
                f"status={result['status_code']} ok={result['ok']} "
                f"time_ms={result['duration_ms']} image={Path(result['image']).name}"
            )

    burst_total_sec = round(time.perf_counter() - burst_started, 3)
    health_after = read_health(base_url)

    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count
    latencies = [float(r["duration_ms"]) for r in results]

    summary = {
        "burst_started_utc": burst_started_utc,
        "base_url": base_url,
        "scenario": args.scenario,
        "parallel_requests": len(selected),
        "success_requests": ok_count,
        "failed_requests": fail_count,
        "total_burst_time_sec": burst_total_sec,
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0,
            "avg": round(statistics.mean(latencies), 2) if latencies else 0.0,
        },
        "health_before": extract_health_brief(health_before),
        "health_after": extract_health_brief(health_after),
    }

    report = {
        "summary": summary,
        "results": sorted(results, key=lambda x: x["image"]),
        "health_before_raw": health_before,
        "health_after_raw": health_after,
    }

    report_path = args.report_path
    if not report_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = str(Path.cwd() / f"parallel_burst_report_{ts}.json")

    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("===== Burst Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[info] Report saved: {report_path}")

    if fail_count:
        print("[warn] Failed requests details:")
        for r in [x for x in results if not x["ok"]]:
            print(f"  image={Path(r['image']).name} status={r['status_code']} error={r['error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
