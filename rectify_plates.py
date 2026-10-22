"""Detect and rectify the most confident licence plate in every image of a folder.

Example:
    python rectify_plates.py --input-dir . --output-dir output
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BOX_PADDING_PX = 10
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    SCRIPT_DIR.parent
    / "runs"
    / "detect"
    / "runs"
    / "yolo26n_car_plate-4"
    / "weights"
    / "best_car_plate.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and rectify the most confident licence plate in each image."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Folder containing JPG, JPEG, or PNG images (default: script folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output",
        help="Folder for crops, rectified images, and report.csv.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to YOLO weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum YOLO confidence in the range 0..1 (default: 0.25).",
    )
    args = parser.parse_args()
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    return args


def read_image(path: Path) -> np.ndarray | None:
    """Read an image even when its path contains non-ASCII characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> bool:
    """Write JPEG through a Unicode-safe path."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True


def unique_jpeg_path(directory: Path, stem: str) -> Path:
    """Return an unused file name without overwriting a previous result."""
    candidate = directory / f"{stem}.jpg"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index}.jpg"
        index += 1
    return candidate


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = points.astype(np.float32)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def rectangle_dimensions(corners: np.ndarray) -> tuple[int, int]:
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    return max(1, int(round(max(top, bottom)))), max(1, int(round(max(left, right))))


def plausible_plate_quad(contour: np.ndarray, crop_area: float) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approximation) != 4 or not cv2.isContourConvex(approximation):
        return None

    corners = order_corners(approximation.reshape(4, 2))
    width, height = rectangle_dimensions(corners)
    area_ratio = cv2.contourArea(approximation) / crop_area
    aspect_ratio = width / height
    if area_ratio < 0.12 or not 1.5 <= aspect_ratio <= 8.5:
        return None
    return corners


def find_plate_corners(crop: np.ndarray) -> np.ndarray | None:
    """Find the strongest plate-like quadrilateral in a detected crop."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = float(crop.shape[0] * crop.shape[1])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        corners = plausible_plate_quad(contour, crop_area)
        if corners is not None:
            candidates.append((cv2.contourArea(contour), corners))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def warp_from_corners(crop: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    width, height = rectangle_dimensions(corners)
    if width < 8 or height < 8:
        return None
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(crop, transform, (width, height), flags=cv2.INTER_CUBIC)


def rotated_fallback(crop: np.ndarray) -> np.ndarray | None:
    """Use a rotated rectangle when no reliable four-corner contour was found."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = crop.shape[0] * crop.shape[1]
    choices: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        if cv2.contourArea(contour) < crop_area * 0.12:
            continue
        rectangle = cv2.minAreaRect(contour)
        width, height = rectangle[1]
        if min(width, height) < 8:
            continue
        ratio = max(width, height) / min(width, height)
        if 1.5 <= ratio <= 8.5:
            choices.append((cv2.contourArea(contour), cv2.boxPoints(rectangle)))
    if not choices:
        return None
    corners = order_corners(max(choices, key=lambda item: item[0])[1])
    return warp_from_corners(crop, corners)


def rectify_crop(crop: np.ndarray) -> tuple[np.ndarray, str]:
    corners = find_plate_corners(crop)
    if corners is not None:
        warped = warp_from_corners(crop, corners)
        if warped is not None:
            return warped, "perspective"
    rotated = rotated_fallback(crop)
    if rotated is not None:
        return rotated, "rotated_fallback"
    return crop, "crop_fallback"


def clamp_crop(image: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
    """Expand a detection by 10 px on every side, clamped to image boundaries."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(float)
    left = max(0, int(np.floor(x1)) - BOX_PADDING_PX)
    top = max(0, int(np.floor(y1)) - BOX_PADDING_PX)
    right = min(width, int(np.ceil(x2)) + BOX_PADDING_PX)
    bottom = min(height, int(np.ceil(y2)) + BOX_PADDING_PX)
    if right <= left or bottom <= top:
        return None, (left, top, right, bottom)
    return image[top:bottom, left:right].copy(), (left, top, right, bottom)


def process_image(model: YOLO, image_path: Path, conf: float, crop_dir: Path, rectified_dir: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "input_file": str(image_path), "confidence": "", "bbox_xyxy": "", "method": "", "crop_file": "", "rectified_file": "", "status": ""
    }
    image = read_image(image_path)
    if image is None:
        record["status"] = "unreadable_image"
        return record
    try:
        result = model.predict(source=image, conf=conf, verbose=False)[0]
    except Exception as error:  # Keep batch processing even when a single inference fails.
        record["status"] = f"inference_error: {error}"
        return record
    if result.boxes is None or len(result.boxes) == 0:
        record["status"] = "no_detection"
        return record

    confidences = result.boxes.conf.cpu().numpy()
    best_index = int(np.argmax(confidences))
    bbox = result.boxes.xyxy[best_index].cpu().numpy()
    crop, clamped_bbox = clamp_crop(image, bbox)
    record["confidence"] = f"{float(confidences[best_index]):.6f}"
    record["bbox_xyxy"] = ",".join(map(str, clamped_bbox))
    if crop is None:
        record["status"] = "invalid_bbox"
        return record

    crop_path = unique_jpeg_path(crop_dir, f"{image_path.stem}_plate")
    rectified_path = unique_jpeg_path(rectified_dir, f"{image_path.stem}_plate")
    if not write_image(crop_path, crop):
        record["status"] = "crop_write_error"
        return record
    rectified, method = rectify_crop(crop)
    rectified_gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    if not write_image(rectified_path, rectified_gray):
        record["status"] = "rectified_write_error"
        record["crop_file"] = str(crop_path)
        return record
    record.update(
        crop_file=str(crop_path), rectified_file=str(rectified_path), method=method, status="ok"
    )
    return record


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"Weights file does not exist: {args.weights}", file=sys.stderr)
        return 2

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    crop_dir = output_dir / "crops"
    rectified_dir = output_dir / "rectified"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No JPG, JPEG, or PNG files found in: {input_dir}")
        return 0

    model = YOLO(str(args.weights))
    fields = ["input_file", "confidence", "bbox_xyxy", "method", "crop_file", "rectified_file", "status"]
    report_path = output_dir / "report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as report:
        writer = csv.DictWriter(report, fieldnames=fields)
        writer.writeheader()
        for image_path in images:
            record = process_image(model, image_path, args.conf, crop_dir, rectified_dir)
            writer.writerow(record)
            print(f"{image_path.name}: {record['status']}" + (f" ({record['method']})" if record["method"] else ""))
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
