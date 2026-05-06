#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx


TARGET_FIELDS = ("owner_code", "registration_number", "check_digit", "type_size_code")


def normalize_text(value: Any) -> str:
    """
    Copy the same normalization idea as release-demo's testing/vlm_eval.py.
    """
    if value is None:
        return ""
    text = str(value).upper()
    # Drop spaces and separators that commonly appear in OCR/VLM outputs.
    return "".join(ch for ch in text if ch.isalnum())


def normalize_fields(fields: Dict[str, Any]) -> Dict[str, str]:
    return {name: normalize_text(fields.get(name, "")) for name in TARGET_FIELDS}


def parse_fields(value: Any) -> Dict[str, str]:
    """
    VLM response may be:
    - dict with keys owner_code/registration_number/...
    - list of {label, text} objects
    - wrapper dict with nested 'result'
    """
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


def load_cases(dataset_path: Path, max_cases: int, default_scenario: str) -> List[dict[str, Any]]:
    dataset_dir = dataset_path.parent.resolve()
    rows = dataset_path.read_text(encoding="utf-8").splitlines()
    cases: List[dict[str, Any]] = []

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
            raise ValueError(f"{case_id}: no image(s) in dataset entry.")
        if isinstance(image_refs, str):
            image_refs = [image_refs]
        if not isinstance(image_refs, list):
            raise ValueError(f"{case_id}: invalid image(s) type.")

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
                "scenario": scenario,
                "metadata": metadata,
                "image_paths": image_paths,
                "expected": expected,
                "has_expected": has_expected,
            }
        )

        if max_cases > 0 and len(cases) >= max_cases:
            break

    if not cases:
        raise RuntimeError("Dataset is empty or no valid rows loaded.")
    return cases


@dataclass
class ImagePayload:
    name: str
    content: bytes
    mime_type: str


def load_image_payload(path: Path) -> ImagePayload:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(str(path))
    return ImagePayload(
        name=path.name,
        content=path.read_bytes(),
        mime_type=mime_type or "application/octet-stream",
    )


def iso_from_expected_or_predicted(fields: Dict[str, str]) -> str:
    return f"{fields.get('owner_code', '')}{fields.get('registration_number', '')}{fields.get('check_digit', '')}".strip()


class ApiSession:
    """
    Minimal copy of release-demo/testing/vlm_eval.py ApiSession logic.
    Uses /auth/login to fetch access token, then calls:
      POST {base_url}/api/v1/recognize-multipart
    """

    def __init__(self, base_url: str, username: str, password: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self._access_token: Optional[str] = None
        self._login_lock = asyncio.Lock()

    async def login(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            f"{self.base_url}/auth/login",
            json={"username": self.username, "password": self.password},
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise RuntimeError("Login succeeded but no access_token in response.")
        self._access_token = token

    async def _ensure_token(self, client: httpx.AsyncClient) -> None:
        if self._access_token:
            return
        async with self._login_lock:
            if not self._access_token:
                await self.login(client)

    async def recognize_multipart(
        self,
        *,
        client: httpx.AsyncClient,
        images: List[ImagePayload],
        metadata_json: str,
    ) -> httpx.Response:
        await self._ensure_token(client)
        headers = {"Authorization": f"Bearer {self._access_token}"}
        files = [("files", (image.name, image.content, image.mime_type)) for image in images]

        response = await client.post(
            f"{self.base_url}/api/v1/recognize-multipart",
            headers=headers,
            data={"metadata": metadata_json},
            files=files,
        )

        # Retry on token expiration (401) like release-demo's test script does.
        if response.status_code == 401:
            await self.login(client)
            headers = {"Authorization": f"Bearer {self._access_token}"}
            response = await client.post(
                f"{self.base_url}/api/v1/recognize-multipart",
                headers=headers,
                data={"metadata": metadata_json},
                files=files,
            )

        return response


def extract_predicted_fields(response_payload: dict[str, Any]) -> Dict[str, str]:
    # release-demo returns: {"result": <scenario_result>, "request_id": ...}
    # In some cases <scenario_result> is wrapped as {"result": <payload>, ...}
    content: Any = response_payload.get("result", response_payload)
    if isinstance(content, dict) and "result" in content:
        content = content["result"]
    return parse_fields(content)


async def async_main(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    cases = load_cases(
        dataset_path=dataset_path,
        max_cases=args.max_samples or 0,
        default_scenario=args.default_scenario,
    )

    timeout = httpx.Timeout(args.timeout)
    sem = asyncio.Semaphore(max(args.concurrency, 1))
    api = ApiSession(args.base_url, args.username, args.password)

    started_at = datetime.now(timezone.utc)
    rows: List[Dict[str, Any]] = []
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=timeout) as client:
        await api.login(client)

        async def run_one(case: dict[str, Any]) -> None:
            async with sem:
                started = time.perf_counter()
                status_code = 0
                error: Optional[str] = None
                predicted = parse_fields({})
                predicted_iso = ""

                try:
                    images = [load_image_payload(p) for p in case["image_paths"]]
                    metadata_json = json.dumps(case["metadata"], ensure_ascii=False)
                    resp = await api.recognize_multipart(
                        client=client, images=images, metadata_json=metadata_json
                    )
                    status_code = resp.status_code
                    if resp.status_code != 200:
                        error = f"http_{resp.status_code}"
                    else:
                        payload = resp.json()
                        predicted = extract_predicted_fields(payload)
                        predicted_iso = iso_from_expected_or_predicted(predicted)
                except Exception as exc:  # noqa: BLE001
                    error = type(exc).__name__

                latency_ms = (time.perf_counter() - started) * 1000.0

                gt_fields: Dict[str, str] = case["expected"]
                gt_iso = iso_from_expected_or_predicted(gt_fields)
                full_match = case["has_expected"] and (predicted.get("owner_code") == gt_fields["owner_code"]
                                                       and predicted.get("registration_number") == gt_fields["registration_number"]
                                                       and predicted.get("check_digit") == gt_fields["check_digit"]
                                                       and predicted.get("type_size_code") == gt_fields["type_size_code"])
                iso_match = case["has_expected"] and predicted_iso.upper() == gt_iso.upper()

                async with lock:
                    rows.append(
                        {
                            "case_id": case["case_id"],
                            "status_code": status_code,
                            "error": error,
                            "latency_ms": latency_ms,
                            "expected": gt_fields,
                            "predicted": predicted,
                            "full_match": full_match if case["has_expected"] else None,
                            "iso_match": iso_match if case["has_expected"] else None,
                        }
                    )

        await asyncio.gather(*(run_one(case) for case in cases))

    ended_at = datetime.now(timezone.utc)

    scored = [r for r in rows if r.get("full_match") is not None]
    scored_iso = [r for r in rows if r.get("iso_match") is not None]

    full_correct = sum(1 for r in scored if r["full_match"])
    iso_correct = sum(1 for r in scored_iso if r["iso_match"])

    # Field-wise accuracies (only where expected exists)
    field_accuracy: Dict[str, Optional[float]] = {}
    for field in TARGET_FIELDS:
        hits = sum(1 for r in scored if r["predicted"].get(field) == r["expected"].get(field))
        field_accuracy[field] = (hits / len(scored)) if scored else None

    status_counts = Counter(r["status_code"] for r in rows)
    avg_latency = (sum(r["latency_ms"] for r in rows) / len(rows)) if rows else None

    return {
        "dataset": str(dataset_path),
        "samples_total": len(rows),
        "samples_scored": len(scored),
        "full_match_accuracy": (full_correct / len(scored)) if scored else 0.0,
        "iso_accuracy": (iso_correct / len(scored_iso)) if scored_iso else 0.0,
        "field_accuracy": field_accuracy,
        "status_codes": {str(k): v for k, v in status_counts.items()},
        "avg_latency_ms": avg_latency,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": ended_at.isoformat(),
        # Keep per-case results for debugging.
        "rows": rows if args.debug_rows else [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VLM via release-demo POST /api/v1/recognize-multipart and compute accuracy."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset JSONL produced by build_release_style_dataset.py.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RELEASE_DEMO_BASE_URL") or "http://localhost:8899",
        help="release-demo base URL (e.g. http://localhost:8899).",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("RELEASE_DEMO_USERNAME") or "admin@example.com",
        help="release-demo username (used for /auth/login).",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("RELEASE_DEMO_PASSWORD") or "admin123@pass!",
        help="release-demo password (used for /auth/login).",
    )
    parser.add_argument(
        "--default-scenario",
        default="extract_number_general",
        help="Fallback scenario name if dataset row doesn't include it.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout seconds.")
    parser.add_argument("--concurrency", type=int, default=2, help="Parallel requests.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--debug-rows", action="store_true", help="Include per-case rows in output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = asyncio.run(async_main(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

