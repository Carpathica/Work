import os
import json
import cv2
import numpy as np
from typing import List, Dict
from ultralytics import YOLO


# =========================
# Константы
# =========================

CONF_KEEP_THRESHOLD = 0.85
ROI_PADDING = 12

ALPHA_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
DIGIT_TO_ALPHA = {v: k for k, v in ALPHA_TO_DIGIT.items()}

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")

# ISO6346 letter values
LETTER_VALUE = {}
val = 10
skip_vals = {11, 22, 33}
for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    while val in skip_vals:
        val += 1
    LETTER_VALUE[ch] = val
    val += 1


# =========================
# ISO6346 checksum
# =========================

def iso6346_check_digit(code10: str) -> int:
    total = 0
    for pos, ch in enumerate(code10[:10]):
        if ch in LETTER_VALUE:
            v = LETTER_VALUE[ch]
        elif ch.isdigit():
            v = int(ch)
        else:
            v = 0
        total += v * (2 ** pos)
    remainder = total % 11
    return 0 if remainder == 10 else remainder


# =========================
# Сортировка символов
# =========================

def sort_characters(chars: List[Dict]) -> List[Dict]:
    """
    Автоопределение:
    - вертикальный текст → сортировка сверху вниз
    - горизонтальный → слева направо
    - 2 строки → кластеризация
    """

    if len(chars) <= 1:
        return chars

    centers_x = []
    centers_y = []

    for c in chars:
        x1, y1, x2, y2 = c["bbox"]
        centers_x.append((x1 + x2) / 2)
        centers_y.append((y1 + y2) / 2)

    x_var = np.var(centers_x)
    y_var = np.var(centers_y)

    # Вертикальный текст
    if y_var > x_var * 1.5:
        return sorted(chars, key=lambda c: c["bbox"][1])

    # Горизонтальный текст
    if x_var > y_var * 1.5:
        return sorted(chars, key=lambda c: c["bbox"][0])

    # Если разброс похож — возможно 2 строки
    # Кластеризация по Y
    ys = np.array(centers_y)
    median_y = np.median(ys)

    top_line = []
    bottom_line = []

    for c, cy in zip(chars, centers_y):
        if cy < median_y:
            top_line.append(c)
        else:
            bottom_line.append(c)

    top_line = sorted(top_line, key=lambda c: c["bbox"][0])
    bottom_line = sorted(bottom_line, key=lambda c: c["bbox"][0])

    return top_line + bottom_line


# =========================
# Коррекция символов
# =========================

def apply_correction(chars: List[Dict], roi_format: str):

    corrected_chars = []

    for i, c in enumerate(chars):
        label = c["label"].upper()
        conf = c["confidence"]

        expected_letter = None

        if roi_format == "cn-4":
            expected_letter = True
        elif roi_format in ["cn-11", "cn-7"]:
            expected_letter = i < 4

        corrected = label

        if expected_letter is True:
            if label in DIGITS and conf < CONF_KEEP_THRESHOLD:
                corrected = DIGIT_TO_ALPHA.get(label, label)

        elif expected_letter is False:
            if label in LETTERS and conf < CONF_KEEP_THRESHOLD:
                corrected = ALPHA_TO_DIGIT.get(label, label)

        c_new = dict(c)
        c_new["corrected_label"] = corrected
        corrected_chars.append(c_new)

    corrected_text = "".join([c["corrected_label"] for c in corrected_chars])

    return corrected_text, corrected_chars


# =========================
# Основной класс
# =========================

class ISO6346Recognizer:

    def __init__(self, roi_model_path, char_model_path, device=None):
        self.roi_model = YOLO(roi_model_path)
        self.char_model = YOLO(char_model_path)

    def detect_rois(self, image):
        result = self.roi_model(image)[0]

        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        rois = []

        for box, conf, cls in zip(boxes, confs, classes):
            rois.append({
                "bbox": box.tolist(),
                "confidence": float(conf),
                "format": self.roi_model.names[int(cls)]
            })

        return rois

    def detect_chars(self, roi_img):
        result = self.char_model(roi_img)[0]

        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        chars = []

        for box, conf, cls in zip(boxes, confs, classes):
            chars.append({
                "bbox": box.tolist(),
                "confidence": float(conf),
                "class_id": int(cls),
                "label": self.char_model.names[int(cls)]
            })

        chars = sort_characters(chars)
        return chars

    def process_image(self, image_path, out_image_path, out_json_path):

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Ошибка загрузки изображения")

        h, w = image.shape[:2]

        result_json = {
            "image": os.path.basename(image_path),
            "rois": []
        }

        rois = self.detect_rois(image)

        for roi in rois:

            x1, y1, x2, y2 = map(int, roi["bbox"])

            # Padding
            x1 = max(0, x1 - ROI_PADDING)
            y1 = max(0, y1 - ROI_PADDING)
            x2 = min(w - 1, x2 + ROI_PADDING)
            y2 = min(h - 1, y2 + ROI_PADDING)

            roi_crop = image[y1:y2, x1:x2].copy()

            chars = self.detect_chars(roi_crop)

            raw_text = "".join([c["label"] for c in chars])

            corrected_text, corrected_chars = apply_correction(chars, roi["format"])

            check_digit = None
            checksum_valid = None

            if roi["format"] == "cn-11" and len(corrected_text) >= 11:
                try:
                    check_digit = int(corrected_text[10])
                    calc = iso6346_check_digit(corrected_text[:10])
                    checksum_valid = (calc == check_digit)
                except:
                    checksum_valid = False

            # абсолютные bbox
            for c in corrected_chars:
                cx1, cy1, cx2, cy2 = c["bbox"]
                c["bbox_abs"] = [
                    x1 + cx1,
                    y1 + cy1,
                    x1 + cx2,
                    y1 + cy2
                ]

            result_json["rois"].append({
                "format": roi["format"],
                "roi_bbox": [x1, y1, x2, y2],
                "confidence": roi["confidence"],
                "raw_text": raw_text,
                "corrected_text": corrected_text,
                "check_digit": check_digit,
                "checksum_valid": checksum_valid,
                "characters": corrected_chars
            })

            # Рисуем ROI
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, corrected_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Рисуем символы
            for c in corrected_chars:
                bx = list(map(int, c["bbox_abs"]))
                cv2.rectangle(image, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 1)
                cv2.putText(image, c["corrected_label"],
                            (bx[0], bx[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1)

        cv2.imwrite(out_image_path, image)

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)

        print("Готово.")
        print("Изображение:", out_image_path)
        print("JSON:", out_json_path)


# =========================
# Запуск
# =========================

if __name__ == "__main__":

    recognizer = ISO6346Recognizer(
        roi_model_path="roi_model.pt",
        char_model_path="char_model.pt"
    )

    recognizer.process_image(
        image_path="input.jpg",
        out_image_path="result.jpg",
        out_json_path="result.json"
    )
