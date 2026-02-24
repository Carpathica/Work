from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import yaml
from PIL import Image

from .schemas import Box

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clean_classes(raw_classes: Iterable[str]) -> List[str]:
    classes: List[str] = []
    for item in raw_classes:
        label = str(item).strip()
        if label:
            classes.append(label)
    return classes


def load_classes(
    dataset_dir: Path,
    provided_classes: Iterable[str],
    classes_file: str | None = None,
) -> List[str]:
    classes = clean_classes(provided_classes)
    if classes:
        return classes

    if classes_file:
        classes_path = resolve_path(dataset_dir, classes_file)
        loaded = _load_classes_file(classes_path)
        if loaded:
            return loaded

    for candidate_name in ("data.yaml", "dataset.yaml", "data.yml", "dataset.yml"):
        candidate = dataset_dir / candidate_name
        if candidate.exists():
            loaded = _load_classes_file(candidate)
            if loaded:
                return loaded

    txt_candidate = dataset_dir / "classes.txt"
    if txt_candidate.exists():
        loaded = _load_classes_file(txt_candidate)
        if loaded:
            return loaded

    return []


def _load_classes_file(path: Path) -> List[str]:
    if not path.exists():
        return []

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        names = data.get("names")
        if isinstance(names, list):
            return clean_classes(names)
        if isinstance(names, dict):
            indexed = []
            for key, value in names.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                indexed.append((idx, str(value)))
            indexed.sort(key=lambda item: item[0])
            return clean_classes(value for _, value in indexed)
        return []

    return clean_classes(path.read_text(encoding="utf-8").splitlines())


def discover_images(dataset_dir: Path) -> List[str]:
    images_root = dataset_dir / "images"
    if images_root.exists() and images_root.is_dir():
        root = images_root
        keep_prefix = True
    else:
        root = dataset_dir
        keep_prefix = False

    images: List[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_dir).as_posix() if keep_prefix else path.relative_to(root).as_posix()
        if rel.startswith("labels/"):
            continue
        images.append(rel)

    images.sort()
    return images


def resolve_path(base: Path, maybe_relative: str) -> Path:
    candidate = Path(maybe_relative).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def ensure_in_base(base: Path, candidate: Path) -> Path:
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError("Path escapes dataset directory.") from exc
    return candidate


def image_absolute_path(dataset_dir: Path, image_rel_path: str) -> Path:
    image_path = (dataset_dir / image_rel_path).resolve()
    return ensure_in_base(dataset_dir, image_path)


def label_relative_path(dataset_dir: Path, image_rel_path: str) -> Path:
    image_rel = Path(image_rel_path)
    has_labels_dir = (dataset_dir / "labels").exists()
    has_images_dir = (dataset_dir / "images").exists()

    if image_rel.parts and image_rel.parts[0] == "images":
        return Path("labels", *image_rel.parts[1:]).with_suffix(".txt")
    if has_images_dir:
        return Path("labels", image_rel_path).with_suffix(".txt")
    if has_labels_dir:
        return Path("labels", image_rel_path).with_suffix(".txt")
    return image_rel.with_suffix(".txt")


def _strip_images_prefix(path: Path) -> Path:
    if path.parts and path.parts[0] == "images":
        return Path(*path.parts[1:])
    return path


def label_absolute_path(
    dataset_dir: Path,
    image_rel_path: str,
    labels_dir: str | None = None,
) -> Path:
    image_rel = _strip_images_prefix(Path(image_rel_path)).with_suffix(".txt")
    if labels_dir:
        labels_root = resolve_path(dataset_dir, labels_dir)
        return (labels_root / image_rel).resolve()

    label_rel = label_relative_path(dataset_dir, image_rel_path)
    return ensure_in_base(dataset_dir, (dataset_dir / label_rel).resolve())


def read_annotations(dataset_dir: Path, image_rel_path: str, labels_dir: str | None = None) -> List[Box]:
    image_path = image_absolute_path(dataset_dir, image_rel_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_rel_path}")

    label_abs = label_absolute_path(dataset_dir, image_rel_path, labels_dir=labels_dir)

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    if not label_abs.exists():
        return []

    boxes: List[Box] = []
    for raw_line in label_abs.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1]) * image_width
            y_center = float(parts[2]) * image_height
            width = float(parts[3]) * image_width
            height = float(parts[4]) * image_height
        except ValueError:
            continue

        x = x_center - width / 2.0
        y = y_center - height / 2.0
        boxes.append(
            Box(
                class_id=max(class_id, 0),
                x=max(0.0, x),
                y=max(0.0, y),
                width=max(0.0, width),
                height=max(0.0, height),
            )
        )
    return boxes


def save_annotations(
    dataset_dir: Path,
    image_rel_path: str,
    boxes: Iterable[Box],
    labels_dir: str | None = None,
) -> Path:
    image_path = image_absolute_path(dataset_dir, image_rel_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_rel_path}")

    with Image.open(image_path) as img:
        image_width, image_height = img.size

    label_abs = label_absolute_path(dataset_dir, image_rel_path, labels_dir=labels_dir)
    label_abs.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for box in boxes:
        width = max(0.0, min(float(box.width), image_width))
        height = max(0.0, min(float(box.height), image_height))
        x = max(0.0, min(float(box.x), image_width - width))
        y = max(0.0, min(float(box.y), image_height - height))
        if width <= 1.0 or height <= 1.0:
            continue

        x_center_norm = (x + width / 2.0) / image_width
        y_center_norm = (y + height / 2.0) / image_height
        width_norm = width / image_width
        height_norm = height / image_height

        line = (
            f"{int(box.class_id)} "
            f"{x_center_norm:.6f} {y_center_norm:.6f} "
            f"{width_norm:.6f} {height_norm:.6f}"
        )
        lines.append(line)

    content = "\n".join(lines)
    if content:
        content += "\n"
    label_abs.write_text(content, encoding="utf-8")
    return label_abs
