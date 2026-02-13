import os
import json
from typing import List, Dict, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- Конфигурация ----------
ROI_CLASSES_EXPECTED = {"cn-11": "cn-11", "cn-7": "cn-7", "cn-4": "cn-4"}
# Порог качества для принятия символа "как есть"
CONF_KEEP_THRESHOLD = 0.85

# Маппинги похожих символов
ALPHA_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
DIGIT_TO_ALPHA = {v: k for k, v in ALPHA_TO_DIGIT.items()}

# Наборы допустимых символов
LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")

# ---------- ISO6346 checksum mapping (from standard) ----------
# В соответствии с ISO 6346 (A=10, B=12, C=13, D=14, E=15, F=16, G=17, H=18, I=19, J=20, K=21, L=23, ...)
# Здесь реализуем строгую мапу как в википедии (исключая 11,22,33).
LETTER_VALUE = {}
# Построим автоматически с учётом правила "начиная с 10, пропуская 11,22,33..."
val = 10
skip_mult = set([11, 22, 33])  # значения, которые нельзя использовать — но при построении это эквивалент "пропуск"
for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    # increase val until it's not in skip_mult
    while val in skip_mult:
        val += 1
    LETTER_VALUE[ch] = val
    val += 1

# ---------- Вспомогательные функции ----------

def iso6346_check_digit(code10: str) -> int:
    """
    code10 -- первые 10 символов контейнерного номера (4 буквы + 6/7 цифр в зависимости)
    Возвращает чек-диджит (0..9)
    Алгоритм: буквы -> значения (см. LETTER_VALUE), цифры оставляем, умножаем на 2^pos (pos от 0 слева),
    суммируем, берем остаток от деления на 11. Если остаток == 10 -> 0.
    """
    if len(code10) < 10:
        raise ValueError("code10 must be length 10")
    total = 0
    for pos, ch in enumerate(code10[:10]):
        if ch.isalpha():
            v = LETTER_VALUE.get(ch.upper(), 0)
        elif ch.isdigit():
            v = int(ch)
        else:
            v = 0
        weight = 1 << pos  # 2**pos
        total += v * weight
    remainder = total % 11
    return 0 if remainder == 10 else remainder

def sort_characters(chars: List[Dict]) -> List[Dict]:
    """
    Сортировка символов: сначала по x (лево->право). Если ширина маленькая и много строк - можно расширить.
    chars: элементы с "bbox": [x1,y1,x2,y2]
    """
    # используем центр по x для сортировки
    for c in chars:
        x1, y1, x2, y2 = c["bbox"]
        c["center_x"] = (x1 + x2) / 2
        c["center_y"] = (y1 + y2) / 2
    chars_sorted = sorted(chars, key=lambda c: (c["center_y"], c["center_x"]))
    # после сортировки удаляем временные поля
    for c in chars_sorted:
        c.pop("center_x", None)
        c.pop("center_y", None)
    return chars_sorted

def apply_confidence_correction(raw_text: str, chars: List[Dict], roi_format: str) -> Tuple[str, List[Dict]]:
    """
    Хитрый фикс похожих символов с учётом confidence и ожидаемого формата.
    - raw_text: склеенный по detected labels
    - chars: список символов (каждый содержит label и confidence)
    - roi_format: строка 'cn-11'|'cn-7'|'cn-4'
    Возвращает (corrected_text, corrected_chars_list)
    """
    corrected_chars = []
    # определяем ожидание: для cn-11 и cn-7 первые 4 символа буквы, остальные цифры; cn-4 — все буквы
    for i, ch_info in enumerate(chars):
        label = ch_info["label"]
        conf = ch_info["confidence"]
        label_up = label.upper()
        expected_letter = False
        if roi_format == "cn-4":
            expected_letter = True
        elif roi_format in ("cn-11", "cn-7"):
            expected_letter = i < 4
        else:
            # если неизвестный формат — сохраняем как есть
            expected_letter = None

        corrected = label_up
        # Если ожидается буква, но получили цифру -> попробовать заменить цифру->букву
        if expected_letter is True and label_up in DIGITS:
            # если уверенность низкая — попробуем заменить
            if conf < CONF_KEEP_THRESHOLD and label_up in DIGIT_TO_ALPHA:
                corrected = DIGIT_TO_ALPHA[label_up]
        # Если ожидается цифра, но получили букву -> попробовать заменить букву->цифра
        elif expected_letter is False and label_up in LETTERS:
            if conf < CONF_KEEP_THRESHOLD and label_up in ALPHA_TO_DIGIT:
                corrected = ALPHA_TO_DIGIT[label_up]
        # Дополнительно: всегда корректируем явные ошибочные символы, если confidence очень низкий
        elif conf < 0.4:
            # слабая детекция — попытка маппинга в обе стороны
            if label_up in ALPHA_TO_DIGIT:
                corrected = ALPHA_TO_DIGIT[label_up]
            elif label_up in DIGIT_TO_ALPHA:
                corrected = DIGIT_TO_ALPHA[label_up]
        # Сохраняем (оставляем исходный confidence — можно перерасчитывать)
        c_new = dict(ch_info)
        c_new["corrected_label"] = corrected
        corrected_chars.append(c_new)

    corrected_text = "".join(c["corrected_label"] for c in corrected_chars)
    return corrected_text, corrected_chars

# ---------- Класс распознавателя ----------

class ISO6346NoOCRRecognizer:
    def __init__(self, roi_model_path: str, char_model_path: str, device: str = None):
        """
        roi_model_path: путь к .pt модели, которая детектит ROI (классы cn-11/cn-7/cn-4 и т.п.)
        char_model_path: модель, детектирующая символы (классы должны быть '0','1',...,'A','B',...)
        device: e.g. 'cpu' или 'cuda:0' (ultralytics сам выберет, если None)
        """
        self.roi_model = YOLO(roi_model_path) if device is None else YOLO(roi_model_path, device=device)
        self.char_model = YOLO(char_model_path) if device is None else YOLO(char_model_path, device=device)

    def detect_rois(self, image: np.ndarray) -> List[Dict]:
        res = self.roi_model(image)[0]
        if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        rois = []
        for box, conf, cls in zip(boxes, confs, classes):
            rois.append({
                "bbox": [float(v) for v in box],  # x1,y1,x2,y2
                "confidence": float(conf),
                "class_id": int(cls),
                "format": str(self.roi_model.names[int(cls)])  # ожидать cn-11/cn-7/cn-4
            })
        return rois

    def detect_chars_in_roi(self, roi_img: np.ndarray) -> List[Dict]:
        """
        Возвращает список символов с bbox относительно ROI (x1,y1,x2,y2), label, confidence, class_id
        """
        res = self.char_model(roi_img)[0]
        if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()
        chars = []
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            label = str(self.char_model.names[int(cls)]).upper()
            chars.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "class_id": int(cls),
                "label": label
            })
        # сортируем символы по центру (по y потом x) — пригодится для мультистрок
        chars = sort_characters(chars)
        return chars

    def process_image(self, image_path: str, out_image_path: str, out_json_path: str, visualize: bool = True):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Не удалось загрузить изображение: " + image_path)

        h_img, w_img = image.shape[:2]
        result = {"image": os.path.basename(image_path), "rois": []}

        rois = self.detect_rois(image)
        for ridx, roi in enumerate(rois):
            x1, y1, x2, y2 = map(int, roi["bbox"])
            # безопасный кроп
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w_img - 1, x2), min(h_img - 1, y2)
            roi_crop = image[y1c:y2c, x1c:x2c].copy()
            chars = self.detect_chars_in_roi(roi_crop)

            # скорректировать метки по ожиданию формата
            raw_text = "".join([c["label"] for c in chars])
            corrected_text, corrected_chars = apply_confidence_correction(raw_text, chars, roi["format"])

            # рассчитываем абсолютные bbox символов (в координатах исходного изображения)
            for c in corrected_chars:
                cx1, cy1, cx2, cy2 = c["bbox"]
                c["bbox_abs"] = [x1c + cx1, y1c + cy1, x1c + cx2, y1c + cy2]

            # валидация ISO6346 (если формат cn-11)
            check_digit = None
            checksum_valid = None
            if roi["format"] == "cn-11" and len(corrected_text) >= 11:
                try:
                    first10 = corrected_text[:10]
                    check_digit = int(corrected_text[10])
                    calc = iso6346_check_digit(first10)
                    checksum_valid = (calc == check_digit)
                except Exception:
                    checksum_valid = False

            # собираем результат для ROI
            roi_entry = {
                "format": roi["format"],
                "roi_bbox": [x1c, y1c, x2c, y2c],
                "roi_confidence": roi["confidence"],
                "raw_text": raw_text,
                "corrected_text": corrected_text,
                "check_digit": check_digit,
                "checksum_valid": checksum_valid,
                "characters": corrected_chars
            }
            result["rois"].append(roi_entry)

            # Визуализация
            if visualize:
                color_roi = (0, 255, 0)
                cv2.rectangle(image, (x1c, y1c), (x2c, y2c), color_roi, 2)
                label_text = f"{roi['format']} {corrected_text}"
                cv2.putText(image, label_text, (x1c, max(15, y1c - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_roi, 2, cv2.LINE_AA)
                # символы
                for c in corrected_chars:
                    bx = list(map(int, c["bbox_abs"]))
                    cv2.rectangle(image, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 1)
                    cv2.putText(image, c["corrected_label"], (bx[0], max(bx[1] - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Сохранение результатов
        if visualize:
            cv2.imwrite(out_image_path, image)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("Сохранено:", out_image_path, out_json_path)

# ---------- Запуск (пример) ----------
if __name__ == "__main__":
    roi_model = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\yolo11_container\YOLO_IDs\ID_YOLO_container\weights\best.pt"      # ваша модель ROI
    char_model = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\yolo11_container\YOLO_Characters\Character_YOLO_container_finetune_large\weights\best.pt"    # ваша модель символов (классы '0'..'9','A'..'Z')
    recognizer = ISO6346NoOCRRecognizer(roi_model, char_model, device=None)

    recognizer.process_image(
        image_path=r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\test_img\1-123808001-OCR-RF-D01_jpg.rf.00dc11c689f9c7178cc65cdecdb7b37d.jpg",
        out_image_path="annotated_result.jpg",
        out_json_path="result_no_ocr.json",
        visualize=True
    )

