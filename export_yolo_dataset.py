#!/usr/bin/env python3
"""Flatten collected YOLO samples into one images/ and labels/ dataset.

The Axis collector saves samples in category/date folders, for example::

    datasets/axis_collected/manual/recognized/2026-08-25/images/camera1_12-00-00.jpg
    datasets/axis_collected/manual/recognized/2026-08-25/labels/camera1_12-00-00.txt

This utility collects every ``images`` directory below the source root and copies
each image together with its same-named YOLO label into a separate output dataset
for each top-level category. For example, ``accepted`` and ``manual`` are exported
to different ``images`` and ``labels`` directories. Original file names are kept;
if a name conflicts, ``_2``, ``_3`` and so on are added, so no images are lost.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "datasets" / "axis_collected"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ExportStats:
    copied: int = 0
    missing_labels: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect dated/category YOLO samples into one images/ and labels/ directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python container_checker/export_yolo_dataset.py
  python container_checker/export_yolo_dataset.py --source-root datasets/axis_collected --output-root datasets/yolo_export
  python container_checker/export_yolo_dataset.py --include-unlabeled

By default, images without a matching .txt label are skipped. Use
--include-unlabeled to add them with an empty label file instead.""",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"Collector dataset root (default: {DEFAULT_SOURCE_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination containing <category>/images and <category>/labels (default: <source-root>/yolo_export).",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Copy images without labels and create empty .txt label files for them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files that already exist in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without copying files.",
    )
    return parser.parse_args()


def image_dirs(source_root: Path, output_root: Path) -> list[Path]:
    """Return input image directories, excluding an output nested under source."""
    dirs: list[Path] = []
    for directory in source_root.rglob("images"):
        if not directory.is_dir():
            continue
        try:
            directory.relative_to(output_root)
        except ValueError:
            dirs.append(directory)
    return sorted(dirs)


def output_stem(image_path: Path) -> str:
    """Build a filesystem-safe version of the original file name."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem).strip("._")
    return safe or "image"


def ensure_output_is_safe(output_root: Path, overwrite: bool) -> None:
    if overwrite:
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output already contains files: {output_root}. Choose another --output-root "
            "or pass --overwrite."
        )


def category_name(source_root: Path, source_images: Path) -> str:
    """Return the top-level collector category for an input images directory."""
    relative = source_images.relative_to(source_root)
    if len(relative.parts) < 2:
        raise ValueError(f"Image directory has no category: {source_images}")
    return relative.parts[0]


def export_dataset(
    source_root: Path,
    output_root: Path,
    *,
    include_unlabeled: bool,
    overwrite: bool,
    dry_run: bool,
) -> ExportStats:
    ensure_output_is_safe(output_root, overwrite)

    copied = 0
    missing_labels = 0
    claimed_names: dict[str, set[str]] = {}
    for source_images in image_dirs(source_root, output_root):
        category = category_name(source_root, source_images)
        output_images = output_root / category / "images"
        output_labels = output_root / category / "labels"
        category_claimed_names = claimed_names.setdefault(category, set())
        source_labels = source_images.parent / "labels"
        for image_path in sorted(source_images.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.is_file() and not include_unlabeled:
                print(f"SKIP (no label): {image_path}")
                missing_labels += 1
                continue

            stem = output_stem(image_path)
            # Covers repeated source names and unusual names which normalize alike.
            original_stem = stem
            number = 2
            while stem in category_claimed_names or (
                not overwrite and (output_images / f"{stem}{image_path.suffix.lower()}").exists()
            ):
                stem = f"{original_stem}_{number}"
                number += 1
            category_claimed_names.add(stem)

            target_image = output_images / f"{stem}{image_path.suffix.lower()}"
            target_label = output_labels / f"{stem}.txt"
            print(f"COPY: {image_path} -> {target_image}")
            if not dry_run:
                output_images.mkdir(parents=True, exist_ok=True)
                output_labels.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, target_image)
                if label_path.is_file():
                    shutil.copy2(label_path, target_label)
                else:
                    target_label.write_text("", encoding="utf-8")
            copied += 1

    return ExportStats(copied=copied, missing_labels=missing_labels)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = (args.output_root or (source_root / "yolo_export")).resolve()

    if not source_root.is_dir():
        print(f"Source root does not exist or is not a directory: {source_root}", file=sys.stderr)
        return 2
    if source_root == output_root:
        print("--output-root must be different from --source-root", file=sys.stderr)
        return 2

    try:
        stats = export_dataset(
            source_root,
            output_root,
            include_unlabeled=args.include_unlabeled,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 2

    action = "Would export" if args.dry_run else "Exported"
    print(f"{action}: {stats.copied} image/label pair(s) to {output_root}")
    if stats.missing_labels:
        print(f"Skipped without labels: {stats.missing_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
