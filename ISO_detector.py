from ultralytics import YOLO
from PIL import Image
import numpy as np
import easyocr
import re
import cv2
import os
import json

print("🖥️ Ejecutando en CPU (GPU deshabilitada)")

# ================================
# MODELOS
# ================================

ids_model = YOLO(
    "C:\\Users\\User\\Desktop\\Projects\\Work\\ISO_6346_rec\\yolo11_container\\YOLO_IDs\\ID_YOLO_container\\weights\\best.pt"
)

# ⚠️ ТВОЯ YOLO-МОДЕЛЬ СИМВОЛОВ ['0'..'9','A'..'Z']
char_model = YOLO(
    "PATH_TO_YOUR_CHAR_MODEL\\weights\\best.pt"
)

ocr_model = easyocr.Reader(['en', 'es'], gpu=False)

video_path = "C:\\Users\\User\\Desktop\\Projects\\Work\\ISO_6346_rec\\yolo11_container\\videos\\label_09158888.mp4"

CONFIDENCE_THRESHOLD_YOLO_IDS = 0.5
CONFIDENCE_THRESHOLD_YOLO_CHARS = 0.4
SCALE_FACTOR = 0.75
SKIP_FRAMES = 3

# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_code(value: str, key: str) -> str:
    if not value:
        return value

    if key in ["cn-11", "cn-4", "code-container"]:
        prefix = (
            value[:4]
            .replace("0", "O")
            .replace("1", "I")
            .replace("5", "S")
            .replace("2", "Z")
            .replace("8", "B")
            .replace("6", "G")
            .replace("4", "A")
            .replace("7", "T")
        )
        return prefix + value[4:]

    return value


# ============================================================
# REGEX
# ============================================================

rules = {
    "code-container": {"attribute": "code-container", "regex": r"^[A-Z]{4}\d{7}$"},
    "cn-11": {"attribute": "cn-11", "regex": r"^[A-Z]{4}\d{7}$"},
    "cn-4": {"attribute": "cn-4", "regex": r"^[A-Z]{4}$"},
    "cn-7": {"attribute": "cn-7", "regex": r"^\d{7}$"},
    "iso-type": {"attribute": "iso-type", "regex": r"^\d{2}[A-Z][A-Z0-9]$"},
}


def parse_detecciones(detecciones, rules):
    parsed = {}
    for key, value in detecciones.items():
        if key not in rules:
            continue

        raw = value["text"] if isinstance(value, dict) else value
        norm = normalize_code(raw, key)
        valid = bool(re.match(rules[key]["regex"], norm))

        parsed[rules[key]["attribute"]] = {
            "raw": raw,
            "normalized": norm,
            "valid": "✔️" if valid else "❌",
        }

    return parsed


# ============================================================
# ISO 6346 CHECK DIGIT
# ============================================================

def calculate_check_digit(container_code: str):
    if not container_code or len(container_code) != 11:
        return None
    if not container_code[:4].isalpha():
        return None

    letter_values = {
        'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17,
        'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25,
        'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32,
        'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
    }

    values = []
    for c in container_code[:10]:
        values.append(letter_values[c] if c.isalpha() else int(c))

    total = sum(v * (2 ** i) for i, v in enumerate(values))
    check_digit = total % 11
    if check_digit == 10:
        check_digit = 0

    return container_code if int(container_code[-1]) == check_digit else None


# ============================================================
# PREDICT IMAGE
# ============================================================

def predict(image: Image.Image):
    detecciones_yolo = {}
    detecciones_easy = {}

    img_gray = image.convert("L")
    img_np = np.array(img_gray)
    img_ready = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

    results_id = ids_model.track(
        img_ready,
        conf=CONFIDENCE_THRESHOLD_YOLO_IDS,
        persist=True,
        verbose=False
    )

    detections = []
    for box in results_id[0].boxes:
        cls = ids_model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        track_id = int(box.id[0]) if box.id is not None else None

        detections.append((cls, conf, (x1, y1, x2, y2), track_id))

    best = {}
    for cls, conf, coords, tid in detections:
        if cls not in best or conf > best[cls][1]:
            best[cls] = (cls, conf, coords, tid)

    img_boxes = Image.fromarray(results_id[0].plot())

    cn4_y, cn7_y, cn11_y = None, None, None
    cn4_e, cn7_e, cn11_e = None, None, None
    track_final = None

    for cls, _, (x1, y1, x2, y2), track_id in best.values():
        crop = image.crop((x1, y1, x2, y2))

        results_char = char_model.predict(
            crop,
            conf=CONFIDENCE_THRESHOLD_YOLO_CHARS,
            verbose=False
        )

        chars = []
        for cbox in results_char[0].boxes:
            char = char_model.names[int(cbox.cls[0])]
            conf = float(cbox.conf[0])
            cx1, cy1, cx2, cy2 = cbox.xyxy[0].tolist()
            chars.append({
                "x": cx1,
                "y": cy1,
                "symbol": char,
                "conf": conf,
                "crop": crop.crop((cx1, cy1, cx2, cy2))
            })

        if cls in ["cn-11", "iso-type"]:
            chars.sort(key=lambda c: c["y"] if crop.height > crop.width else c["x"])
        else:
            chars.sort(key=lambda c: c["x"])

        text_yolo = "".join(c["symbol"] for c in chars)
        conf_yolo = np.mean([c["conf"] for c in chars]) if chars else 0.0

        detecciones_yolo[cls] = {
            "text": text_yolo,
            "confidence": conf_yolo,
            "track_id": track_id
        }

        # ---------- EasyOCR ----------
        ocr_img = crop
        if chars:
            total_w = sum(c["crop"].width for c in chars)
            max_h = max(c["crop"].height for c in chars)
            recon = Image.new("RGB", (total_w, max_h))
            xoff = 0
            for c in chars:
                recon.paste(c["crop"], (xoff, 0))
                xoff += c["crop"].width
            ocr_img = recon

        ocr = ocr_model.readtext(np.array(ocr_img), detail=1)
        if ocr:
            txt = "".join(r[1] for r in ocr).upper()
            txt = re.sub(r'[^A-Z0-9]', '', txt)
            conf = np.mean([r[2] for r in ocr])
            detecciones_easy[cls] = {"text": txt, "confidence": conf}

        if cls == "cn-11":
            cn11_y, cn11_e = text_yolo, detecciones_easy.get(cls, {}).get("text")
            track_final = track_id
        if cls == "cn-4":
            cn4_y, cn4_e = text_yolo, detecciones_easy.get(cls, {}).get("text")
        if cls == "cn-7":
            cn7_y, cn7_e = text_yolo, detecciones_easy.get(cls, {}).get("text")

    if cn11_y:
        detecciones_yolo["code-container"] = {
            "text": cn11_y,
            "confidence": detecciones_yolo["cn-11"]["confidence"],
            "track_id": track_final
        }
    elif cn4_y and cn7_y:
        detecciones_yolo["code-container"] = {
            "text": cn4_y + cn7_y,
            "confidence": np.mean([
                detecciones_yolo["cn-4"]["confidence"],
                detecciones_yolo["cn-7"]["confidence"]
            ]),
            "track_id": track_final
        }

    if cn11_e:
        detecciones_easy["code-container"] = {"text": cn11_e}
    elif cn4_e and cn7_e:
        detecciones_easy["code-container"] = {"text": cn4_e + cn7_e}

    parsed_y = parse_detecciones(detecciones_yolo, rules)
    parsed_e = parse_detecciones(detecciones_easy, rules)

    valid_y = calculate_check_digit(parsed_y["code-container"]["normalized"]) if "code-container" in parsed_y else None
    valid_e = calculate_check_digit(parsed_e["code-container"]["normalized"]) if "code-container" in parsed_e else None

    final_code = valid_y or valid_e

    return img_boxes, {
        "codigo_final_id": final_code,
        "codigo_final_track": track_final
    }
