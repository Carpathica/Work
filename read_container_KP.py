from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from read_container import (
    DEFAULT_MERGE_IOU,
    Detection,
    _collect_images,
    _greedy_iou_suppress_same_label,
    _normalize_text,
    _pick_cluster_representatives_for_reading,
    cluster_detections_into_lines,
    format_container_output,
    iso6346_check_valid,
)

if TYPE_CHECKING:
    from ultralytics import YOLO


DEFAULT_KP_SOURCE = Path(
    r"C:\Users\ander\Desktop\Projects\Work\ISO_6346_recognition\datasets\dataset_for_benchmark\KP"
)



RowKind = Literal["owner", "digits", "type", "mixed"]


@dataclass(slots=True)
class ParsedRow:
    kind: RowKind
    text: str
    dets: list[Detection]




def _line_text(line: list[Detection]) -> str:
    return _normalize_text("".join(d.label for d in line))


def _classify_four_chars(text: str) -> RowKind | None:
    t = _normalize_text(text)

    if len(t) != 4:
        return None

    if t.isalpha():
        return "owner"

    # ISO 6346 type/size code: two digits, then a letter, then a digit
    # (for example, 22G1 or 28U5). This avoids numeric serial fragments.
    if t[:2].isdigit() and t[2].isalpha() and t[3].isdigit():
        return "type"

    return None




def _detections_from_result_kp(
    r0,
    model_names: dict | list | None,
    merge_iou: float | None,
) -> list[Detection]:

    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = r0.names or model_names

    xyxy_all = boxes.xyxy.cpu().numpy()
    cls_all = boxes.cls.cpu().numpy().astype(int)
    conf_all = boxes.conf.cpu().numpy()

    xyxy_list = []
    conf_list = []
    label_list = []

    for i in range(len(boxes)):
        x1, y1, x2, y2 = xyxy_all[i]

        xyxy_list.append((float(x1), float(y1), float(x2), float(y2)))
        conf_list.append(float(conf_all[i]))

        cls_id = int(cls_all[i])

        if isinstance(names, dict):
            label_list.append(str(names.get(cls_id, cls_id)))
        else:
            label_list.append(str(names[cls_id]))

    indices = list(range(len(xyxy_list)))

    if merge_iou is not None and len(indices) > 1:
        indices = _greedy_iou_suppress_same_label(
            xyxy_list,
            conf_list,
            label_list,
            merge_iou,
        )

        indices = _pick_cluster_representatives_for_reading(
            indices,
            xyxy_list,
            conf_list,
            label_list,
            merge_iou,
        )

    out: list[Detection] = []

    for i in indices:
        x1, y1, x2, y2 = xyxy_list[i]

        out.append(
            Detection(
                label_list[i],
                float((y1 + y2) * 0.5),
                float((x1 + x2) * 0.5),
                (float(x1), float(y1), float(x2), float(y2)),
            )
        )

    return out



def _parse_line(line: list[Detection]) -> list[ParsedRow]:

    text = _line_text(line)

    if text.isdigit():
        return [ParsedRow("digits", text, line)]

    out: list[ParsedRow] = []

    # Поиск owner/type через sliding window
    for i in range(len(line) - 3):

        win = line[i : i + 4]
        four = _line_text(win)

        kind = _classify_four_chars(four)

        if kind is None:
            continue

        out.append(ParsedRow(kind, four, win))

        # всё после owner — цифровой хвост
        if kind == "owner":

            tail = line[i + 4 :]
            if tail:

                digits = "".join(
                    c for c in _line_text(tail) if c.isdigit()
                )

                if digits:
                    out.append(
                        ParsedRow("digits", digits, tail)
                    )

    if not out:
        out.append(ParsedRow("mixed", text, line))

    return out


def _parse_rows(lines: list[list[Detection]]) -> list[ParsedRow]:

    out: list[ParsedRow] = []

    for line in lines:
        out.extend(_parse_line(line))

    return out



def _build_candidates(owner: str, digits: str) -> list[str]:

    owner = _normalize_text(owner)
    digits = "".join(c for c in _normalize_text(digits) if c.isdigit())

    if len(owner) != 4 or not owner.isalpha():
        return []

    out: list[str] = []

    # windows длиной 7
    for i in range(max(1, len(digits) - 6)):

        chunk = digits[i : i + 7]

        if len(chunk) == 7:
            out.append(owner + chunk)


    # fallback
    if len(digits) < 7:
        out.append(owner + digits)

    # unique
    seen = set()
    unique = []

    for s in out:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


def _candidate_score(text: str) -> int:

    score = 0

    if len(text) == 11:
        score += 10

    if iso6346_check_valid(text):
        score += 100

    score -= abs(11 - len(text))

    return score




def read_container_kp_from_detections(
    dets: list[Detection],
) -> tuple[str, bool, list[Detection], str, str | None]:

    if not dets:
        return "", False, [], "kp_empty", None

    lines = cluster_detections_into_lines(dets)
    rows = _parse_rows(lines)

    owners = [r for r in rows if r.kind == "owner"]
    digits_rows = [r for r in rows if r.kind == "digits"]
    type_rows = [r for r in rows if r.kind == "type"]

    digits_joined = "".join(r.text for r in digits_rows)

    best_text = ""
    best_score = -10_000
    best_owner: list[Detection] = []

    for owner_row in owners:

        candidates = _build_candidates(
            owner_row.text,
            digits_joined,
        )

        for cand in candidates:

            score = _candidate_score(cand)

            if score > best_score:
                best_score = score
                best_text = cand
                best_owner = owner_row.dets

    check_ok = (
        len(best_text) == 11
        and iso6346_check_valid(best_text)
    )

    ordered: list[Detection] = []

    if type_rows:
        ordered.extend(type_rows[0].dets)

    ordered.extend(best_owner)

    for row in digits_rows:
        ordered.extend(row.dets)

    # layout
    has_owner = bool(owners)
    has_digits = bool(digits_rows)
    has_type = bool(type_rows)

    if has_owner and has_digits and has_type:
        layout = "kp_owner+digits+type"
    elif has_owner and has_digits:
        layout = "kp_owner+digits"
    elif has_owner and has_type:
        layout = "kp_owner+type"
    elif has_digits:
        layout = "kp_digits_only"
    else:
        layout = "kp_rows"

    type_code = type_rows[0].text if type_rows else None

    return (
        best_text,
        check_ok,
        ordered,
        layout,
        type_code,
    )




def predict_container_kp_with_layout(
    model: "YOLO",
    image_path: Path,
    conf: float = 0.15,
    iou: float = 0.45,
    agnostic_nms: bool = False,
    max_det: int = 300,
    merge_iou: float | None = DEFAULT_MERGE_IOU,
) -> tuple[str, bool, list[Detection], str, str | None]:

    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        verbose=False,
    )

    if not results:
        return "", False, [], "kp_empty", None

    dets = _detections_from_result_kp(
        results[0],
        model.names if hasattr(model, "names") else None,
        merge_iou,
    )

    if not dets:
        return "", False, [], "kp_empty", None

    return read_container_kp_from_detections(dets)



def main() -> int:

    parser = argparse.ArgumentParser(
        description="KP container reader.",
    )

    parser.add_argument("--weights", "-w", type=Path, required=True)

    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=DEFAULT_KP_SOURCE,
    )

    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)

    parser.add_argument(
        "--merge-iou",
        type=float,
        default=DEFAULT_MERGE_IOU,
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--window-scale", type=float, default=4.0)

    args = parser.parse_args()

    merge_iou = None if args.merge_iou <= 0 else args.merge_iou

    paths = _collect_images(args.source)

    if not paths:
        print(f"Нет изображений: {args.source}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))

    if args.debug:

        from read_container_debug import run_interactive_debug_viewer

        run_interactive_debug_viewer(
            model,
            paths,
            predict_fn=predict_container_kp_with_layout,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            merge_iou=merge_iou,
            window_scale=args.window_scale,
        )

        return 0

    for image_path in paths:

        t0 = time.perf_counter()

        text, _check_ok, _ordered, _layout, size_type = (
            predict_container_kp_with_layout(
                model,
                image_path,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                merge_iou=merge_iou,
            )
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print(
            format_container_output(
                text,
                size_type,
                elapsed_ms,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())