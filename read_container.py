# Чтение номера контейнера ISO 6346: YOLO → порядок символов → проверка → CLI/отладка.
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple

if TYPE_CHECKING:
    from ultralytics import YOLO


# Расширения файлов изображений для аргумента --source
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Имя окна OpenCV в режиме --debug (см. read_container_debug.py)
DEBUG_WINDOW_NAME = "ISO6346 debug"

# После NMS YOLO: удаление одного бокса из пары с IoU выше порога (по большей уверенности). 0 — выкл
DEFAULT_MERGE_IOU = 0.35

# Бокс удаляется, если расстояние до ближайшего соседа > factor × медианный размер символа (фильтрация выбросов)
OUTLIER_MAX_NEAREST_NEIGHBOR_FACTOR = 4.0

# Код типа контейнера ISO
_AUX_SIZE_TYPE_4 = re.compile(r"^[0-9]{2}[A-Z0-9][0-9]$")
# Вариант с двумя буквами в конце
_AUX_SIZE_TYPE_4_LETTERS = re.compile(r"^[0-9]{2}[A-Z]{1,2}$")
# ISO 6346: 3 буквы владельца + категория оборудования U/J/Z + 6 цифр + check digit
_ISO_OWNER_SERIAL = re.compile(r"^[A-Z]{3}[UJZ][0-9]{7}$")


# Проверить, похожа ли строка на код типа контейнера (2–5 символов, не только цифры)
def _is_aux_size_type_code(text: str) -> bool:
    t = text.strip().upper()
    if len(t) < 2 or len(t) > 5:
        return False
    if not any(c.isalpha() for c in t):
        return False
    if len(t) == 4 and _AUX_SIZE_TYPE_4.fullmatch(t):
        return True
    if len(t) == 4 and _AUX_SIZE_TYPE_4_LETTERS.fullmatch(t):
        return True
    if len(t) in (2, 3) and re.fullmatch(r"[0-9A-Z]+", t):
        return any(c.isalpha() for c in t)
    return False


# Порог кластеризации строк по cy (доля медианной высоты бокса)
LINE_CLUSTER_Y_FACTOR = 0.52


# Построить таблицу числовых значений букв по ISO 6346 (пропуск кратных 11) 
def _build_iso_letter_values() -> dict[str, int]:
    out: dict[str, int] = {}
    value = 10
    forbidden = {11, 22, 33}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        while value in forbidden:
            value += 1
        out[letter] = value
        value += 1
    return out


# Таблица числовых значений букв ISO 6346 (пропуск кратных 11)
_ISO_LETTER_VALUE: dict[str, int] = _build_iso_letter_values()

# Приоритет раскладок при равном score (больше — предпочтительнее)
_LAYOUT_TIEBREAK_PRIORITY: dict[str, int] = {
    "vertical": 0,
    "horizontal_one": 1,
    "split_owner_four": 2,
    "horizontal_three": 3,
    "horizontal_two": 4,
    "vertical_two_columns": 5,
}


# Один распознанный символ: метка класса, центр и bbox в пикселях
class Detection(NamedTuple):
    label: str
    cy: float
    cx: float
    xyxy: tuple[float, float, float, float]


# Результат чтения: номер, проверка ISO, порядок боксов, тип контейнера, раскладка
class ContainerRead(NamedTuple):
    primary_number: str
    check_ok: bool
    ordered: list[Detection]
    size_type_code: str | None
    layout: str


class ScoredDetection(NamedTuple):
    label: str
    cy: float
    cx: float
    xyxy: tuple[float, float, float, float]
    conf: float


class CameraRead(NamedTuple):
    image_path: Path
    primary_number: str
    check_ok: bool
    ordered: list[ScoredDetection]
    size_type_code: str | None
    layout: str
    char_scores: list[tuple[str, float]]


class DualRead(NamedTuple):
    primary_number: str
    check_ok: bool
    size_type_code: str | None
    layout: str
    fusion: str
    camera1: CameraRead
    camera2: CameraRead


# Числовое значение символа для расчёта контрольной цифры ISO 6346
def _char_value(c: str) -> int:
    c = c.upper()
    if c.isdigit():
        return int(c)
    if c in _ISO_LETTER_VALUE:
        return _ISO_LETTER_VALUE[c]
    raise ValueError(f"Недопустимый символ для ISO 6346: {c!r}")


# Нормализовать строку: верхний регистр, без пробелов
def _normalize_text(text: str) -> str:
    return text.strip().upper().replace(" ", "")


# Контрольная цифра ISO 6346 по первым 10 символам номера (0–9)
def iso6346_check_digit(first_ten: str) -> int:
    if len(first_ten) != 10:
        raise ValueError("Нужно ровно 10 символов")
    total = sum(_char_value(first_ten[i]) * (2**i) for i in range(10))
    r = total % 11
    return 0 if r == 10 else r


# True, если 11-й символ совпадает с контрольной цифрой ISO 6346
def iso6346_check_valid(full: str) -> bool:
    full = _normalize_text(full)
    if len(full) != 11:
        return False
    try:
        expected = iso6346_check_digit(full[:10])
    except ValueError:
        return False
    return full[10].isdigit() and int(full[10]) == expected


def iso6346_number_format_valid(full: str) -> bool:
    return bool(_ISO_OWNER_SERIAL.fullmatch(_normalize_text(full)))


# Сортировка сверху вниз; при близком y — слева направо
def sort_vertical_reading_order(detections: Iterable[Detection]) -> list[Detection]:
    return sorted(detections, key=lambda d: (d.cy, d.cx))


# --- Дополнительные раскладки ---

# Отделить код типа контейнера в соседней колонке по cx от 11-символьного номера
def split_distant_right_auxiliary(
    dets: list[Detection],
) -> tuple[list[Detection], str | None]:
    if len(dets) < 12:
        return dets, None

    s = sorted(dets, key=lambda d: d.cx)
    widths = [max(0.0, d.xyxy[2] - d.xyxy[0]) for d in s]
    med_w = max(1.0, _median(widths))
    min_cx_gap = med_w * 0.2

    best: tuple[list[Detection], str, int] | None = None  # core dets, aux code, score

    for split_i in range(len(s) - 1):
        group_low = s[: split_i + 1]
        group_high = s[split_i + 1 :]
        gap_cx = float(group_high[0].cx - group_low[-1].cx)

        orientations = (
            (group_high, group_low, "aux_left"),
            (group_low, group_high, "aux_right"),
        )
        for main_group, aux_group, _tag in orientations:
            ma, aa = len(main_group), len(aux_group)
            if aa < 2 or aa > 5 or ma < 10:
                continue

            main_ordered = sort_vertical_reading_order(main_group)
            aux_ordered = sort_vertical_reading_order(aux_group)
            main_txt = _join_labels(main_ordered)
            aux_txt = _join_labels(aux_ordered, normalized=True)

            aux_ok = _is_aux_size_type_code(aux_txt)
            iso_ok = _is_valid_iso_text(main_txt)
            if not aux_ok and not iso_ok:
                continue
            if not aux_ok and gap_cx < min_cx_gap:
                continue
            if aux_ok and not iso_ok and not (len(main_txt) == 11):
                continue

            score = 0
            if iso_ok:
                score += 300
            if aux_ok:
                score += 200
            if len(main_txt) == 11:
                score += 30
            if re.fullmatch(r"[A-Z]{4}[0-9]{7}", main_txt):
                score += 150
            score += min(int(max(gap_cx, 0.0)), 80)

            if best is None or score > best[2]:
                best = (main_ordered, aux_txt, score)

    if best is None:
        return dets, None
    return best[0], best[1]




# Метка символа в верхнем регистре
def _norm_label(d: Detection) -> str:
    return _normalize_text(d.label)


# Склеить метки детекций в строку
def _join_labels(dets: Iterable[Detection], *, normalized: bool = False) -> str:
    if normalized:
        return "".join(_norm_label(d) for d in dets)
    return "".join(d.label for d in dets)


# Строка длиной 11 и проходит проверку ISO 6346
def _is_valid_iso_text(text: str) -> bool:
    text = _normalize_text(text)
    return len(text) == 11 and iso6346_number_format_valid(text) and iso6346_check_valid(text)


# Высота и ширина бокса в пикселях
def _det_hw(d: Detection) -> tuple[float, float]:
    x1, y1, x2, y2 = d.xyxy
    return max(0.0, y2 - y1), max(0.0, x2 - x1)


# Медиана списка; пустой список - 0
def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    a = sorted(xs)
    m = len(a) // 2
    return float(a[m]) if len(a) % 2 else 0.5 * (a[m - 1] + a[m])


# Разбить детекции на горизонтальные строки по близости cy
def cluster_detections_into_lines(
    dets: list[Detection], line_y_factor: float = LINE_CLUSTER_Y_FACTOR
) -> list[list[Detection]]:
    if not dets:
        return []
    heights = [_det_hw(d)[0] for d in dets]
    med_h = max(1.0, _median(heights))
    thresh = max(5.0, med_h * line_y_factor)
    by_cy = sorted(dets, key=lambda d: d.cy)
    lines: list[list[Detection]] = []
    for d in by_cy:
        placed = False
        for line in lines:
            my = sum(x.cy for x in line) / len(line)
            if abs(d.cy - my) < thresh:
                line.append(d)
                placed = True
                break
        if not placed:
            lines.append([d])
    for line in lines:
        line.sort(key=lambda d: d.cx)
    lines.sort(key=lambda ln: sum(d.cy for d in ln) / len(ln))
    return lines


# Сортировка слева направо (одна строка)
def sort_horizontal_one_line(dets: Iterable[Detection]) -> list[Detection]:
    return sorted(dets, key=lambda d: d.cx)


# 4 буквы владельца слева и 7 символов справа (раскладка split_owner_four)
def try_split_owner_four(dets: list[Detection]) -> tuple[list[Detection], str] | None:
    if len(dets) < 8:
        return None
    s = sort_horizontal_one_line(dets)
    widths = [_det_hw(d)[1] for d in s]
    med_w = max(1.0, _median(widths))
    gaps: list[tuple[float, int]] = []
    for i in range(len(s) - 1):
        gap = float(s[i + 1].xyxy[0] - s[i].xyxy[2])
        gaps.append((gap, i))
    if not gaps:
        return None
    best_gap, split_i = max(gaps, key=lambda t: t[0])
    if best_gap < med_w * 0.85:
        return None
    left, right = s[: split_i + 1], s[split_i + 1 :]
    if len(left) != 4 or len(right) != 7:
        return None
    owner = _join_labels(left, normalized=True)
    if not re.fullmatch(r"[A-Z]{4}", owner):
        return None
    return left + right, owner


# Собрать 11-символьный номер и код типа контейнера из двух строк
def _pick_primary_iso_from_two_lines(top: str, bottom: str) -> tuple[str, str | None]:
    a, b = top, bottom
    if len(b) > len(a):
        a, b = b, a
    code_line = b if b else None
    if len(a) == 11:
        return a, code_line
    if len(a) > 11:
        return a[:11], (a[11:] + b) if b else a[11:]
    if len(a) + len(b) == 11:
        return a + b, None
    return a + b, code_line


# Две вертикальные колонки в одной плоскости (vertical_two_columns)
def _reading_from_vertical_two_columns(
    dets: list[Detection],
) -> tuple[str, bool, list[Detection], str | None] | None:
    if len(dets) < 12:
        return None

    lines = cluster_detections_into_lines(dets)
    paired_rows = [ln for ln in lines if len(ln) >= 2]
    if len(paired_rows) < 4 or len(paired_rows) / len(lines) < 0.35:
        return None

    row_gaps: list[float] = []
    for ln in paired_rows:
        ln_s = sorted(ln, key=lambda d: d.cx)
        row_gaps.append(float(ln_s[-1].cx - ln_s[0].cx))
    widths = [_det_hw(d)[1] for d in dets]
    if _median(row_gaps) < max(1.0, _median(widths)) * 0.45:
        return None

    left_col: list[Detection] = []
    right_col: list[Detection] = []
    left_cxs: list[float] = []
    right_cxs: list[float] = []

    for ln in lines:
        ln_s = sorted(ln, key=lambda d: d.cx)
        if len(ln_s) >= 2:
            left_col.append(ln_s[0])
            right_col.append(ln_s[-1])
            left_cxs.append(ln_s[0].cx)
            right_cxs.append(ln_s[-1].cx)
            continue

        d = ln_s[0]
        split_cx = 0.5 * (_median(left_cxs) + _median(right_cxs)) if left_cxs and right_cxs else d.cx
        if right_col and len(right_col) >= 4:
            left_col.append(d)
            left_cxs.append(d.cx)
        elif left_col and len(left_col) >= 11:
            right_col.append(d)
            right_cxs.append(d.cx)
        elif d.cx < split_cx:
            left_col.append(d)
            left_cxs.append(d.cx)
        else:
            right_col.append(d)
            right_cxs.append(d.cx)

    left_s = _join_labels(left_col, normalized=True)
    right_s = _join_labels(right_col, normalized=True)
    if not left_s or not right_s:
        return None

    best: tuple[int, str, str | None, list[Detection], list[Detection]] | None = None
    for main_col, aux_col, main_s, aux_s in (
        (left_col, right_col, left_s, right_s),
        (right_col, left_col, right_s, left_s),
    ):
        if not _is_valid_iso_text(main_s):
            continue
        aux_stripped = _normalize_text(aux_s)
        if aux_stripped and not _is_aux_size_type_code(aux_stripped):
            continue
        score = 300
        if aux_stripped and _is_aux_size_type_code(aux_stripped):
            score += 200
        if re.fullmatch(r"[A-Z]{4}[0-9]{7}", main_s):
            score += 150
        if best is None or score > best[0]:
            best = (
                score,
                main_s,
                aux_stripped or None,
                sort_vertical_reading_order(main_col),
                sort_vertical_reading_order(aux_col),
            )

    if best is None:
        return None
    _, primary, st, main_ordered, aux_ordered = best
    ordered = main_ordered + aux_ordered
    return primary, True, ordered, st


# Две строки: номер и код типа контейнера (horizontal_two)
def _reading_from_horizontal_two_lines(
    dets: list[Detection],
) -> tuple[str, bool, list[Detection], str | None] | None:
    lines = cluster_detections_into_lines(dets)
    if len(lines) != 2:
        return None
    s0 = _join_labels(lines[0], normalized=True)
    s1 = _join_labels(lines[1], normalized=True)
    primary, code_line = _pick_primary_iso_from_two_lines(s0, s1)
    primary = _normalize_text(primary)
    st = _normalize_text(code_line or "") or None
    if not _is_valid_iso_text(primary):
        return None
    if st and not _is_aux_size_type_code(st):
        return None
    ordered = _ordered_two_lines(lines)
    return primary, True, ordered, st


# Оценка варианта раскладки; чем больше — тем лучше
def _layout_candidate_score(
    layout: str,
    text: str,
    check_ok: bool,
    size_type_line: str | None,
) -> int:
    score = 0
    if check_ok:
        score += 1000
    if len(text) == 11:
        score += 50
    if size_type_line and _is_aux_size_type_code(size_type_line):
        score += {
            "vertical_two_columns": 450,
            "horizontal_two": 400,
            "horizontal_three": 350,
        }.get(layout, 0)
    score += {
        "split_owner_four": 200,
        "horizontal_one": 100,
        "vertical": 0,
    }.get(layout, 0)
    if re.fullmatch(r"[A-Z]{4}[0-9]{7}", text):
        score += 150
    return score


# Собрать номер из трёх строк (horizontal_three)
def _pick_primary_iso_from_three_lines(t0: str, t1: str, t2: str) -> tuple[str, str | None]:
    head = t0 + t1
    if len(head) == 11:
        return head, t2 or None
    if len(head) > 11:
        return head[:11], (head[11:] + t2) if t2 else head[11:]
    full = t0 + t1 + t2
    if len(full) == 11:
        return full, None
    if len(full) > 11:
        return full[:11], full[11:]
    return full, None


# Итоговый код типа контейнера из раскладки и/или split по cx
def _merge_size_type_code(
    layout: str,
    from_layout: str | None,
    from_right_split: str | None,
) -> str | None:
    a = _normalize_text(from_layout or "") or None
    b = _normalize_text(from_right_split or "") or None
    if layout in ("horizontal_two", "horizontal_three", "vertical_two_columns"):
        return a or b
    return b or a


# Порядок боксов при чтении двух строк
def _ordered_two_lines(lines: list[list[Detection]]) -> list[Detection]:
    return [d for ln in lines for d in ln]


# Порядок боксов при чтении трёх строк
def _ordered_three_lines(lines: list[list[Detection]]) -> list[Detection]:
    return [d for ln in lines[:3] for d in ln]


# Вертикальное чтение набора боксов без доп. раскладок
def _reading_from_vertical_core(core: list[Detection]) -> tuple[str, bool, list[Detection]]:
    ordered = sort_vertical_reading_order(core)
    text = _join_labels(ordered)
    ok = _is_valid_iso_text(text)
    return text, ok, ordered


# Список кандидатов чтения для всех раскладок
def _collect_layout_candidates(
    dets: list[Detection], core: list[Detection]
) -> list[tuple[str, str, bool, list[Detection], str | None]]:
    out: list[tuple[str, str, bool, list[Detection], str | None]] = []

    vcols = _reading_from_vertical_two_columns(dets)
    if vcols is not None:
        text, ok, ordered, st = vcols
        out.append(("vertical_two_columns", text, ok, ordered, st))

    tv, ok_v, ov = _reading_from_vertical_core(core)
    out.append(("vertical", tv, ok_v, ov, None))

    o_h = sort_horizontal_one_line(core)
    th = _join_labels(o_h)
    out.append(("horizontal_one", th, _is_valid_iso_text(th), o_h, None))

    sp = try_split_owner_four(dets)
    if sp is not None:
        o_sp, _owner = sp
        ts = _join_labels(o_sp)
        out.append(("split_owner_four", ts, _is_valid_iso_text(ts), o_sp, None))

    lines = cluster_detections_into_lines(dets)
    if len(lines) == 2:
        s0 = _join_labels(lines[0], normalized=True)
        s1 = _join_labels(lines[1], normalized=True)
        primary, code_line = _pick_primary_iso_from_two_lines(s0, s1)
        primary = _normalize_text(primary)
        st_line = _normalize_text(code_line or "") or None
        o2 = _ordered_two_lines(lines)
        out.append(
            (
                "horizontal_two",
                primary,
                _is_valid_iso_text(primary),
                o2,
                st_line,
            )
        )
    if len(lines) >= 3:
        strs = [_join_labels(ln, normalized=True) for ln in lines[:3]]
        primary, code_line = _pick_primary_iso_from_three_lines(strs[0], strs[1], strs[2])
        primary = _normalize_text(primary)
        extra = _normalize_text("".join(_join_labels(ln, normalized=True) for ln in lines[3:]))
        st_line = _normalize_text(code_line or "") or None
        if extra:
            st_line = (st_line + extra) if st_line else extra
        o3 = _ordered_three_lines(lines)
        out.append(
            (
                "horizontal_three",
                primary,
                _is_valid_iso_text(primary),
                o3,
                st_line or None,
            )
        )

    return out


# Ключ сортировки для выбора лучшего кандидата
def _candidate_rank(cand: tuple[str, str, bool, list[Detection], str | None]) -> tuple[int, int, int]:
    layout, text, check_ok, _ordered, st_lines = cand
    score = _layout_candidate_score(layout, text, check_ok, st_lines)
    return score, _LAYOUT_TIEBREAK_PRIORITY.get(layout, -1), len(text)


# Выбрать лучшее чтение среди всех раскладок
def pick_best_reading_with_extra_layouts(
    dets: list[Detection],
) -> tuple[str, bool, list[Detection], str, str | None]:
    if not dets:
        return "", False, [], "vertical", None

    core, size_from_split = split_distant_right_auxiliary(dets)
    cands = _collect_layout_candidates(dets, core)
    best_pool = [c for c in cands if c[2]] or cands
    layout, text, check_ok, ordered, st_lines = max(best_pool, key=_candidate_rank)
    st = _merge_size_type_code(layout, st_lines, size_from_split)
    return text, check_ok, ordered, layout, st


# IoU двух bbox в формате xyxy
def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


# Убрать дубли одного символа с высоким IoU (жадное подавление)
def _greedy_iou_suppress_same_label(
    xyxy_list: list[tuple[float, float, float, float]],
    conf_list: list[float],
    label_list: list[str],
    iou_thresh: float,
) -> list[int]:
    n = len(xyxy_list)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: conf_list[i], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [
            j
            for j in order
            if _iou_xyxy(xyxy_list[i], xyxy_list[j]) <= iou_thresh
            or label_list[i] != label_list[j]
        ]
    return keep


# Сгруппировать индексы боксов с IoU выше порога
def _cluster_indices_by_iou(indices: list[int], xyxy_list: list[tuple[float, float, float, float]], iou_thresh: float) -> list[list[int]]:
    parent = {i: i for i in indices}

# Найти корень в системе непересекающихся множеств (union-find)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

# Объединить два множества (union-find)
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ai, i in enumerate(indices):
        for j in indices[ai + 1 :]:
            if _iou_xyxy(xyxy_list[i], xyxy_list[j]) > iou_thresh:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in indices:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# По одному боксу на кластер; при D/O — перебор по ISO
def _pick_cluster_representatives_for_reading(
    indices: list[int],
    xyxy_list: list[tuple[float, float, float, float]],
    conf_list: list[float],
    label_list: list[str],
    iou_thresh: float,
) -> list[int]:
    clusters = _cluster_indices_by_iou(indices, xyxy_list, iou_thresh)
    if not clusters:
        return []

    choice_lists: list[list[int]] = []
    for cluster in clusters:
        labels_in = {label_list[i] for i in cluster}
        if len(labels_in) == 1:
            choice_lists.append([max(cluster, key=lambda i: conf_list[i])])
        else:
            choice_lists.append(sorted(cluster, key=lambda i: conf_list[i], reverse=True))

    if all(len(c) == 1 for c in choice_lists):
        return [c[0] for c in choice_lists]

    best_pick: list[int] | None = None
    best_key = (-1, -1.0)

    for combo in product(*choice_lists):
        tmp: list[Detection] = []
        for i in combo:
            x1, y1, x2, y2 = xyxy_list[i]
            tmp.append(
                Detection(
                    label=label_list[i],
                    cy=float((y1 + y2) * 0.5),
                    cx=float((x1 + x2) * 0.5),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        ordered = sort_vertical_reading_order(tmp)
        text = "".join(d.label for d in ordered)
        ok = _is_valid_iso_text(text)
        conf_sum = sum(conf_list[i] for i in combo)
        key = (1 if ok else 0, conf_sum)
        if key > best_key:
            best_key = key
            best_pick = list(combo)

    if best_pick is not None:
        return best_pick
    return [max(cluster, key=lambda i: conf_list[i]) for cluster in clusters]


# Оставить крупнейшую связную группу по расстоянию между центрами
def filter_spatial_outlier_detections(
    dets: list[Detection],
    *,
    max_nn_factor: float = OUTLIER_MAX_NEAREST_NEIGHBOR_FACTOR,
) -> list[Detection]:
    if len(dets) <= 3:
        return dets

    sizes = [max(_det_hw(d)[0], _det_hw(d)[1]) for d in dets]
    med_size = max(1.0, _median(sizes))
    max_nn_dist = max_nn_factor * med_size
    n = len(dets)
    parent = list(range(n))

# Найти корень в системе непересекающихся множеств (union-find)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

# Объединить два множества (union-find)
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(dets[i].cx - dets[j].cx, dets[i].cy - dets[j].cy) <= max_nn_dist:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    largest = max(groups.values(), key=len)
    if len(largest) >= max(3, (n + 1) // 2):
        return [dets[i] for i in largest]
    return dets


# Преобразовать результат YOLO в список Detection (merge IoU, фильтр выбросов)
def _detections_from_result(
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

    xyxy_list: list[tuple[float, float, float, float]] = []
    conf_list: list[float] = []
    label_list: list[str] = []
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
        indices = _greedy_iou_suppress_same_label(xyxy_list, conf_list, label_list, merge_iou)
        indices = _pick_cluster_representatives_for_reading(indices, xyxy_list, conf_list, label_list, merge_iou)

    dets: list[Detection] = []
    for i in indices:
        label = label_list[i]
        x1, y1, x2, y2 = xyxy_list[i]
        cx = float((x1 + x2) * 0.5)
        cy = float((y1 + y2) * 0.5)
        dets.append(Detection(label=label, cy=cy, cx=cx, xyxy=(float(x1), float(y1), float(x2), float(y2))))
    return filter_spatial_outlier_detections(dets)


def _to_detection(d: ScoredDetection) -> Detection:
    return Detection(d.label, d.cy, d.cx, d.xyxy)


def _scored_from_detection(d: Detection, conf: float) -> ScoredDetection:
    return ScoredDetection(d.label, d.cy, d.cx, d.xyxy, conf)


def _scored_detections_from_result(
    r0,
    model_names: dict | list | None,
    merge_iou: float | None,
) -> list[ScoredDetection]:
    boxes = r0.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = r0.names or model_names
    xyxy_all = boxes.xyxy.cpu().numpy()
    cls_all = boxes.cls.cpu().numpy().astype(int)
    conf_all = boxes.conf.cpu().numpy()

    xyxy_list: list[tuple[float, float, float, float]] = []
    conf_list: list[float] = []
    label_list: list[str] = []
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
        indices = _greedy_iou_suppress_same_label(xyxy_list, conf_list, label_list, merge_iou)
        indices = _pick_cluster_representatives_for_reading(indices, xyxy_list, conf_list, label_list, merge_iou)

    scored: list[ScoredDetection] = []
    for i in indices:
        x1, y1, x2, y2 = xyxy_list[i]
        scored.append(
            ScoredDetection(
                label=label_list[i],
                cy=float((y1 + y2) * 0.5),
                cx=float((x1 + x2) * 0.5),
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
                conf=conf_list[i],
            )
        )

    filtered = filter_spatial_outlier_detections([_to_detection(d) for d in scored])
    by_xyxy = {d.xyxy: d for d in scored}
    return [_scored_from_detection(d, by_xyxy[d.xyxy].conf) for d in filtered]


def _match_scored_detections(
    ordered: list[Detection],
    scored: list[ScoredDetection],
) -> list[ScoredDetection]:
    by_xyxy = {d.xyxy: d for d in scored}
    out: list[ScoredDetection] = []
    used: set[int] = set()
    for d in ordered:
        hit = by_xyxy.get(d.xyxy)
        if hit is not None:
            out.append(hit)
            used.add(id(hit))
            continue

        best: ScoredDetection | None = None
        best_dist = 1e18
        for s in scored:
            if id(s) in used:
                continue
            if _normalize_text(s.label) != _normalize_text(d.label):
                continue
            dist = (s.cx - d.cx) ** 2 + (s.cy - d.cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = s
        if best is not None:
            out.append(best)
            used.add(id(best))
        else:
            out.append(_scored_from_detection(d, 0.0))
    return out


def _char_scores_for_primary(
    primary: str,
    ordered: Iterable[ScoredDetection],
) -> list[tuple[str, float]]:
    primary = _normalize_text(primary)
    if not primary:
        return []

    chars: list[str] = []
    confs: list[float] = []
    for det in ordered:
        label = _normalize_text(det.label)
        for ch in label:
            chars.append(ch)
            confs.append(det.conf)

    joined = "".join(chars)
    start = joined.find(primary)
    if start >= 0:
        end = start + len(primary)
        return [(chars[i], confs[i]) for i in range(start, end)]

    if len(chars) >= len(primary):
        return [(chars[i], confs[i]) for i in range(len(primary))]

    return [(primary[i], 0.0) for i in range(len(primary))]


def _camera_primary_confidence(read: CameraRead) -> float:
    return sum(conf for _ch, conf in read.char_scores[:11])


def _score_fused_text(text: str, conf_sum: float) -> tuple[int, float, int]:
    normalized = _normalize_text(text)
    score = 0
    if _is_valid_iso_text(normalized):
        score += 1000
    if len(normalized) == 11:
        score += 50
    else:
        score -= abs(11 - len(normalized))
    if re.fullmatch(r"[A-Z]{4}[0-9]{7}", normalized):
        score += 150
    return score, conf_sum, len(normalized)


def _fuse_disputed_positions(
    scores1: list[tuple[str, float]],
    scores2: list[tuple[str, float]],
    *,
    max_positions: int = 11,
) -> tuple[str, bool, str]:
    s1 = list(scores1)
    s2 = list(scores2)
    n = min(max(len(s1), len(s2)), max_positions)
    if n == 0:
        return "", False, "empty"

    while len(s1) < n:
        s1.append(("", 0.0))
    while len(s2) < n:
        s2.append(("", 0.0))

    choices: list[list[tuple[str, float]]] = []
    disputed = 0
    for i in range(n):
        c1, cf1 = s1[i]
        c2, cf2 = s2[i]
        opts: dict[str, float] = {}
        if c1:
            opts[c1] = max(opts.get(c1, 0.0), cf1)
        if c2:
            opts[c2] = max(opts.get(c2, 0.0), cf2)
        if not opts:
            opts[""] = 0.0
        if len(opts) > 1:
            disputed += 1
        choices.append(list(opts.items()))

    best_text = ""
    best_key = (-10_000, -1.0, -1)
    for combo in product(*choices):
        text = _normalize_text("".join(ch for ch, _conf in combo))
        conf_sum = sum(conf for _ch, conf in combo)
        key = _score_fused_text(text, conf_sum)
        if key > best_key:
            best_key = key
            best_text = text

    ok = _is_valid_iso_text(best_text)
    return best_text[:11], ok, "char_fusion" if disputed else "char_agree"


def _clean_size_type_code(value: str | None) -> str | None:
    raw = _normalize_text(value or "")
    if not raw:
        return None
    if _is_aux_size_type_code(raw):
        return raw
    for size in (4, 3, 2):
        for start in range(0, len(raw) - size + 1):
            candidate = raw[start : start + size]
            if _is_aux_size_type_code(candidate):
                return candidate
    return raw


def _pick_size_type_code(cam1: CameraRead, cam2: CameraRead, winner: int | None) -> str | None:
    codes = {
        1: _clean_size_type_code(cam1.size_type_code),
        2: _clean_size_type_code(cam2.size_type_code),
    }

    if winner in (1, 2) and codes[winner] and _is_aux_size_type_code(codes[winner]):
        return codes[winner]
    for idx in (1, 2):
        code = codes[idx]
        if code and _is_aux_size_type_code(code):
            return code
    if winner in (1, 2) and codes[winner]:
        return codes[winner]
    return codes[1] or codes[2]


# Детекции - лучшее чтение с перебором раскладок
def _result_to_ordered_detections(
    r0,
    model_names: dict | list | None,
    merge_iou: float | None,
) -> tuple[str, bool, list[Detection], str, str | None]:
    dets = _detections_from_result(r0, model_names, merge_iou)
    if not dets:
        return "", False, [], "vertical", None
    return pick_best_reading_with_extra_layouts(dets)


# Распознать контейнер; вернуть номер, раскладку и тип контейнера
def predict_container_with_layout(
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
        return "", False, [], "vertical", None
    text, ok, ordered, layout, size_type = _result_to_ordered_detections(
        results[0],
        model.names if hasattr(model, "names") else None,
        merge_iou,
    )
    if not iso6346_number_format_valid(text):
        return "", False, ordered, layout, size_type
    return text, ok, ordered, layout, size_type


# Распознать контейнер (без имени раскладки в кортеже)
def predict_container(
    model: "YOLO",
    image_path: Path,
    conf: float = 0.25,
    iou: float = 0.45,
    agnostic_nms: bool = False,
    max_det: int = 300,
    merge_iou: float | None = DEFAULT_MERGE_IOU,
) -> tuple[str, bool, list[Detection], str | None]:
    text, ok, ordered, _layout, aux = predict_container_with_layout(
        model,
        image_path,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        merge_iou=merge_iou,
    )
    return text, ok, ordered, aux


# Распознать контейнер; результат в виде ContainerRead
def predict_container_read(
    model: "YOLO",
    image_path: Path,
    conf: float = 0.15,
    iou: float = 0.45,
    agnostic_nms: bool = False,
    max_det: int = 300,
    merge_iou: float | None = DEFAULT_MERGE_IOU,
) -> ContainerRead:
    text, ok, ordered, layout, aux = predict_container_with_layout(
        model,
        image_path,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        merge_iou=merge_iou,
    )
    return ContainerRead(text, ok, ordered, aux, layout)


def predict_camera_read(
    model: "YOLO",
    image_path: Path,
    conf: float = 0.15,
    iou: float = 0.45,
    agnostic_nms: bool = False,
    max_det: int = 300,
    merge_iou: float | None = DEFAULT_MERGE_IOU,
) -> CameraRead:
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        verbose=False,
    )
    if not results:
        return CameraRead(image_path, "", False, [], None, "empty", [])

    scored = _scored_detections_from_result(
        results[0],
        model.names if hasattr(model, "names") else None,
        merge_iou,
    )
    if not scored:
        return CameraRead(image_path, "", False, [], None, "empty", [])

    text, check_ok, ordered, layout, size_type = pick_best_reading_with_extra_layouts(
        [_to_detection(d) for d in scored]
    )
    scored_ordered = _match_scored_detections(ordered, scored)
    char_scores = _char_scores_for_primary(text, scored_ordered)
    if not iso6346_number_format_valid(text):
        return CameraRead(image_path, "", False, scored_ordered, size_type, layout, char_scores)
    return CameraRead(image_path, text, check_ok, scored_ordered, size_type, layout, char_scores)


def fuse_dual_camera_reads(cam1: CameraRead, cam2: CameraRead) -> DualRead:
    t1 = _normalize_text(cam1.primary_number)
    t2 = _normalize_text(cam2.primary_number)

    if t1 and t1 == t2:
        winner = 1 if cam1.check_ok or not cam2.check_ok else 2
        return DualRead(
            t1,
            cam1.check_ok or cam2.check_ok,
            _pick_size_type_code(cam1, cam2, winner),
            cam1.layout if winner == 1 else cam2.layout,
            "both_agree",
            cam1,
            cam2,
        )

    if cam1.check_ok and not cam2.check_ok:
        return DualRead(t1, True, _pick_size_type_code(cam1, cam2, 1), cam1.layout, "cam1_valid", cam1, cam2)

    if cam2.check_ok and not cam1.check_ok:
        return DualRead(t2, True, _pick_size_type_code(cam1, cam2, 2), cam2.layout, "cam2_valid", cam1, cam2)

    if cam1.check_ok and cam2.check_ok:
        c1 = _camera_primary_confidence(cam1)
        c2 = _camera_primary_confidence(cam2)
        if c1 >= c2:
            return DualRead(t1, True, _pick_size_type_code(cam1, cam2, 1), cam1.layout, "cam1_confidence", cam1, cam2)
        return DualRead(t2, True, _pick_size_type_code(cam1, cam2, 2), cam2.layout, "cam2_confidence", cam1, cam2)

    fused, ok, fusion = _fuse_disputed_positions(cam1.char_scores, cam2.char_scores)
    if ok:
        size_type = _pick_size_type_code(cam1, cam2, None)
        layout = cam1.layout if _camera_primary_confidence(cam1) >= _camera_primary_confidence(cam2) else cam2.layout
        return DualRead(fused, True, size_type, layout, fusion, cam1, cam2)

    if _score_fused_text(t1, _camera_primary_confidence(cam1)) >= _score_fused_text(
        t2, _camera_primary_confidence(cam2)
    ):
        return DualRead(t1, False, _pick_size_type_code(cam1, cam2, 1), cam1.layout, "cam1_fallback", cam1, cam2)
    return DualRead(t2, False, _pick_size_type_code(cam1, cam2, 2), cam2.layout, "cam2_fallback", cam1, cam2)


def predict_dual_container(
    model: "YOLO",
    cam1_path: Path,
    cam2_path: Path,
    conf: float = 0.15,
    iou: float = 0.45,
    agnostic_nms: bool = False,
    max_det: int = 300,
    merge_iou: float | None = DEFAULT_MERGE_IOU,
) -> DualRead:
    cam1 = predict_camera_read(
        model,
        cam1_path,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        merge_iou=merge_iou,
    )
    cam2 = predict_camera_read(
        model,
        cam2_path,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        merge_iou=merge_iou,
    )
    return fuse_dual_camera_reads(cam1, cam2)


# Загрузить веса YOLO и распознать одно изображение
def read_container_from_image(
    image_path: Path,
    weights: Path,
    conf: float = 0.15,
    iou: float = 0.45,
    agnostic_nms: bool = False,
    max_det: int = 300,
    merge_iou: float | None = DEFAULT_MERGE_IOU,
) -> tuple[str, bool, list[Detection], str | None]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return predict_container(
        model,
        image_path,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        merge_iou=merge_iou,
    )


# То же, что read_container_from_image, но ContainerRead
def read_container_read_from_image(
    image_path: Path,
    weights: Path,
    conf: float = 0.15,
    iou: float = 0.45,
    agnostic_nms: bool = False,
    max_det: int = 300,
    merge_iou: float | None = DEFAULT_MERGE_IOU,
) -> ContainerRead:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return predict_container_read(
        model,
        image_path,
        conf=conf,
        iou=iou,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        merge_iou=merge_iou,
    )


# Разбить 11-символьный номер на поля JSON (owner, registration, check_digit, type_size).
def format_container_result(
    primary: str,
    size_type_code: str | None = None,
) -> list[dict[str, str]]:
    s = _normalize_text(primary)
    type_size = _normalize_text(size_type_code or "")
    return [
        {"label": "owner_code", "text": s[:4] if len(s) >= 4 else ""},
        {"label": "registration_number", "text": s[4:10] if len(s) >= 10 else s[4:]},
        {"label": "check_digit", "text": s[10] if len(s) >= 11 else ""},
        {"label": "type_size_code", "text": type_size},
    ]


# Собрать итоговый JSON для stdout: result и время распознавания (мс).
def format_container_output(
    primary: str,
    size_type_code: str | None,
    elapsed_ms: float,
) -> str:
    payload = {
        "result": format_container_result(primary, size_type_code),
        "elapsed_ms": round(elapsed_ms, 2),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# Список изображений из файла или каталога для --source
def _collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source.resolve()] if source.suffix.lower() in IMAGE_SUFFIXES else []
    if source.is_dir():
        return sorted(
            p.resolve()
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def _collect_single_image(source: Path) -> Path | None:
    paths = _collect_images(source)
    if len(paths) == 1:
        return paths[0]
    return None


def _camera_pair_in_dir(folder: Path) -> tuple[Path, Path] | None:
    images = _collect_images(folder)
    if len(images) != 2:
        return None

    cam1 = next((p for p in images if "camera1" in p.name.lower()), None)
    cam2 = next((p for p in images if "camera2" in p.name.lower()), None)
    if cam1 is not None and cam2 is not None:
        return cam1, cam2
    return images[0], images[1]


def _collect_dual_image_pairs(source: Path) -> list[tuple[Path, Path, Path]]:
    source = source.resolve()
    if not source.is_dir():
        return []

    direct = _camera_pair_in_dir(source)
    if direct is not None:
        cam1, cam2 = direct
        return [(source, cam1, cam2)]

    pairs: list[tuple[Path, Path, Path]] = []
    for sub in sorted(source.iterdir(), key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name)):
        if not sub.is_dir():
            continue
        found = _camera_pair_in_dir(sub)
        if found is None:
            continue
        cam1, cam2 = found
        pairs.append((sub, cam1, cam2))
    return pairs


# Точка входа CLI: табличный вывод или --debug
def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO + вертикальный порядок + ISO 6346.")
    parser.add_argument("--weights", "-w", type=Path, required=True, help="Путь к .pt модели YOLO.")
    parser.add_argument(
        "--source",
        "-s",
        "--source1",
        "-s1",
        dest="source1",
        type=Path,
        required=True,
        help="Файл изображения или папка с изображениями.",
    )
    parser.add_argument(
        "--source2",
        "-s2",
        type=Path,
        default=None,
        help="Второй файл изображения для режима двух камер.",
    )
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=DEFAULT_MERGE_IOU,
        help=(
            "Слияние дублей после YOLO: одинаковые символы с большим IoU; "
            "разные буквы на одном месте (D/O) — выбор по контрольной цифре ISO. "
            f"По умолчанию {DEFAULT_MERGE_IOU}. 0 — отключить."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Окно OpenCV: n/Space вперёд, p/b назад, q/Esc выход.",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=4.0,
        help="Масштаб окна в режиме --debug (относительно размера кадра).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Печатать детали по каждой камере в stderr в режиме двух камер.",
    )
    args = parser.parse_args()

    merge_iou: float | None = None if args.merge_iou <= 0 else args.merge_iou

    paths = _collect_images(args.source1)
    dual_inputs: tuple[Path, Path] | None = None
    dual_pair_jobs: list[tuple[Path, Path, Path]] = []
    if args.source2 is not None:
        cam1_path = _collect_single_image(args.source1)
        cam2_path = _collect_single_image(args.source2)
        if cam1_path is None:
            print(f"Нужен ровно один файл изображения для source1: {args.source1}", file=sys.stderr)
            return 1
        if cam2_path is None:
            print(f"Нужен ровно один файл изображения для source2: {args.source2}", file=sys.stderr)
            return 1
        dual_inputs = (cam1_path, cam2_path)
        dual_pair_jobs = [(cam1_path.parent, cam1_path, cam2_path)]
    elif args.debug:
        dual_pair_jobs = _collect_dual_image_pairs(args.source1)

    if not paths and not dual_pair_jobs:
        print(f"Нет изображений: {args.source1}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(args.weights.resolve()))

    if args.debug and dual_pair_jobs:
        from read_container_debug import run_dual_interactive_debug_viewer

        run_dual_interactive_debug_viewer(
            model,
            dual_pair_jobs,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            merge_iou=merge_iou,
            window_scale=args.window_scale,
        )
        return 0

    if dual_inputs is not None:
        cam1_path, cam2_path = dual_inputs
        t0 = time.perf_counter()
        dual = predict_dual_container(
            model,
            cam1_path,
            cam2_path,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            merge_iou=merge_iou,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if args.verbose:
            print(
                f"camera1={dual.camera1.primary_number!r} "
                f"ok={dual.camera1.check_ok} layout={dual.camera1.layout} "
                f"type={dual.camera1.size_type_code!r}",
                file=sys.stderr,
            )
            print(
                f"camera2={dual.camera2.primary_number!r} "
                f"ok={dual.camera2.check_ok} layout={dual.camera2.layout} "
                f"type={dual.camera2.size_type_code!r}",
                file=sys.stderr,
            )
            print(
                f"fusion={dual.fusion} result={dual.primary_number!r} "
                f"ok={dual.check_ok} layout={dual.layout} type={dual.size_type_code!r}",
                file=sys.stderr,
            )

        print(format_container_output(dual.primary_number, dual.size_type_code, elapsed_ms))
        return 0

    if args.debug:
        from read_container_debug import run_interactive_debug_viewer

        run_interactive_debug_viewer(
            model,
            paths,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            merge_iou=merge_iou,
            window_scale=args.window_scale,
        )
        return 0

    for image_path in paths:
        t0 = time.perf_counter()
        text, _check_ok, _ordered, _layout, size_type = predict_container_with_layout(
            model,
            image_path,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            merge_iou=merge_iou,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(format_container_output(text, size_type, elapsed_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
