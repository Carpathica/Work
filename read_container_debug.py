# OpenCV-отладка для read_container: просмотр рамок и порядка чтения.
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np

from read_container import (
    DEBUG_WINDOW_NAME,
    CameraRead,
    Detection,
    DualRead,
    predict_container_with_layout,
    predict_dual_container,
)

if TYPE_CHECKING:
    from ultralytics import YOLO

# (primary, check_ok, ordered, layout, size_type_code)
PredictWithLayoutFn = Callable[..., tuple[str, bool, list[Detection], str, str | None]]


# Нарисовать рамки, порядок символов и подписи на кадре.
def draw_debug_overlay(
    image_bgr: np.ndarray,
    ordered: list[Detection],
    title: str,
    footer: str = "",
) -> np.ndarray:
    vis = image_bgr.copy()
    h, _w = vis.shape[:2]
    ordered_str = "".join(d.label for d in ordered)

    for i, d in enumerate(ordered):
        x1, y1, x2, y2 = map(int, d.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            vis,
            str(i + 1),
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        ordered_str,
        (10, min(h - 10, 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        title,
        (10, min(h - 10, 80)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    if footer:
        cv2.putText(
            vis,
            footer,
            (10, min(h - 10, 115)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


# Создать и масштабировать окно OpenCV.
def _ensure_debug_window(window_name: str, vis_width: int, vis_height: int, scale: float) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    dw = max(1, int(vis_width * scale))
    dh = max(1, int(vis_height * scale))
    cv2.resizeWindow(window_name, dw, dh)


# Ожидать нажатие клавиши с поддержкой стрелок.
def _wait_key_debug() -> int:
    fn = getattr(cv2, "waitKeyEx", cv2.waitKey)
    return int(fn(0) & 0xFFFFFFFF)


# Проверить, запрошен ли переход к следующему кадру.
def _key_is_next(k: int) -> bool:
    if k in (ord("n"), ord("N"), ord(" "), 13):
        return True
    return k in (2555904, 65363)


# Проверить, запрошен ли переход к предыдущему кадру.
def _key_is_prev(k: int) -> bool:
    if k in (ord("p"), ord("P"), ord("b"), ord("B"), 8):
        return True
    return k in (2424832, 65361)


# Проверить, запрошен ли выход из просмотра.
def _key_is_quit(k: int) -> bool:
    return k in (ord("q"), ord("Q"), 27)


def _ordered_as_detections(read: CameraRead) -> list[Detection]:
    return [Detection(d.label, d.cy, d.cx, d.xyxy) for d in read.ordered]


def _resize_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image
    scale = target_h / max(1, h)
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _camera_debug_panel(
    image_bgr: np.ndarray,
    read: CameraRead,
    *,
    camera_label: str,
) -> np.ndarray:
    iso_note = "ISO checksum OK" if read.check_ok else (
        "checksum mismatch / invalid" if len(read.primary_number) == 11 else f"len={len(read.primary_number)}"
    )
    st = f" size_type_code={read.size_type_code!r}" if read.size_type_code else " size_type_code=None"
    title = f"{camera_label}: {read.image_path.name} | {iso_note} | {read.layout}{st}"
    footer = f"primary={read.primary_number!r} | check_ok={read.check_ok}"
    return draw_debug_overlay(image_bgr, _ordered_as_detections(read), title, footer=footer)


def draw_dual_debug_overlay(
    cam1_bgr: np.ndarray,
    cam1_read: CameraRead,
    cam2_bgr: np.ndarray,
    cam2_read: CameraRead,
    dual: DualRead,
    *,
    pair_title: str,
    nav_hint: str,
) -> np.ndarray:
    target_h = max(cam1_bgr.shape[0], cam2_bgr.shape[0])
    left = _resize_to_height(_camera_debug_panel(cam1_bgr, cam1_read, camera_label="cam1"), target_h)
    right = _resize_to_height(_camera_debug_panel(cam2_bgr, cam2_read, camera_label="cam2"), target_h)

    gap = 8
    separator = np.full((target_h, gap, 3), 40, dtype=np.uint8)
    combined = np.hstack([left, separator, right])

    iso_note = "ISO checksum OK" if dual.check_ok else (
        "checksum mismatch / invalid" if len(dual.primary_number) == 11 else f"len={len(dual.primary_number)}"
    )
    header = (
        f"{pair_title} | fused={dual.primary_number!r} | {iso_note} | "
        f"fusion={dual.fusion} | layout={dual.layout} | type={dual.size_type_code!r}"
    )
    footer = (
        f"{nav_hint} | cam1={cam1_read.primary_number!r} cam2={cam2_read.primary_number!r} "
        f"| check_ok={dual.check_ok}"
    )

    bar_h = 72
    canvas = np.full((combined.shape[0] + bar_h, combined.shape[1], 3), 24, dtype=np.uint8)
    canvas[bar_h:, :] = combined
    cv2.putText(
        canvas,
        header,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        footer,
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def run_dual_interactive_debug_viewer(
    model: "YOLO",
    pairs: list[tuple[Path, Path, Path]],
    *,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: float | None,
    window_scale: float,
) -> None:
    if not pairs:
        return

    idx = 0
    first = True
    nav_hint = "n/Space/Enter/Right: next pair | p/b/Back/Left: prev | q/Esc: quit"
    read_fail_streak = 0
    window_name = f"{DEBUG_WINDOW_NAME} dual"

    while True:
        pair_dir, cam1_path, cam2_path = pairs[idx]
        dual = predict_dual_container(
            model,
            cam1_path,
            cam2_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            merge_iou=merge_iou,
        )

        cam1_bgr = cv2.imread(str(cam1_path))
        cam2_bgr = cv2.imread(str(cam2_path))
        if cam1_bgr is None or cam2_bgr is None:
            missing = []
            if cam1_bgr is None:
                missing.append(str(cam1_path))
            if cam2_bgr is None:
                missing.append(str(cam2_path))
            print(f"(debug) не прочитать файл: {', '.join(missing)}", file=sys.stderr)
            read_fail_streak += 1
            if read_fail_streak >= len(pairs):
                break
            idx = (idx + 1) % len(pairs)
            continue
        read_fail_streak = 0

        pair_title = f"pair {pair_dir.name} ({idx + 1}/{len(pairs)})"
        vis = draw_dual_debug_overlay(
            cam1_bgr,
            dual.camera1,
            cam2_bgr,
            dual.camera2,
            dual,
            pair_title=pair_title,
            nav_hint=nav_hint,
        )
        h, w = vis.shape[:2]
        if first:
            _ensure_debug_window(window_name, w, h, window_scale)
            first = False
        cv2.imshow(window_name, vis)

        k = _wait_key_debug()
        if _key_is_quit(k):
            break
        if _key_is_next(k):
            idx = (idx + 1) % len(pairs)
        elif _key_is_prev(k):
            idx = (idx - 1) % len(pairs)

    cv2.destroyAllWindows()


# Интерактивный просмотр: n/→ вперёд, p/← назад, q/Esc выход.
def run_interactive_debug_viewer(
    model: "YOLO",
    paths: list[Path],
    *,
    predict_fn: PredictWithLayoutFn | None = None,
    conf: float,
    iou: float,
    max_det: int,
    merge_iou: float | None,
    window_scale: float,
) -> None:
    if not paths:
        return

    idx = 0
    first = True
    nav_hint = "n/Space/Enter/Right: next | p/b/Back/Left: prev | q/Esc: quit"
    read_fail_streak = 0

    predict = predict_fn or predict_container_with_layout

    while True:
        image_path = paths[idx]
        text, check_ok, ordered, layout, size_type = predict(
            model,
            image_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            merge_iou=merge_iou,
        )
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"(debug) не прочитать файл: {image_path}", file=sys.stderr)
            read_fail_streak += 1
            if read_fail_streak >= len(paths):
                break
            idx = (idx + 1) % len(paths)
            continue
        read_fail_streak = 0

        iso_note = "ISO checksum OK" if check_ok else (
            "checksum mismatch / invalid" if len(text) == 11 else f"len={len(text)}"
        )
        st = f" size_type_code={size_type!r}" if size_type else " size_type_code=None"
        title = f"{image_path.name}  ({idx + 1}/{len(paths)})  |  {iso_note}  |  {layout}{st}"
        footer = f"{nav_hint}  |  primary={text!r}  |  size_type_code={size_type!r}  |  check_ok={check_ok}"
        vis = draw_debug_overlay(bgr, ordered, title, footer=footer)
        h, w = vis.shape[:2]
        if first:
            _ensure_debug_window(DEBUG_WINDOW_NAME, w, h, window_scale)
            first = False
        cv2.imshow(DEBUG_WINDOW_NAME, vis)

        k = _wait_key_debug()
        if _key_is_quit(k):
            break
        if _key_is_next(k):
            idx = (idx + 1) % len(paths)
        elif _key_is_prev(k):
            idx = (idx - 1) % len(paths)

    cv2.destroyAllWindows()
