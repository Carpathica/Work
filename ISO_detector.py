from ultralytics import YOLO
from PIL import Image
import numpy as np
import easyocr
import re
import cv2
import os
import json

# ================================
# Настройки
# ================================
SCALE_FACTOR = 0.75
SKIP_FRAMES = 3
CONFIDENCE_THRESHOLD_YOLO_IDS = 0.5
CONFIDENCE_THRESHOLD_YOLO_CHARS = 0.4

# Пути к моделям YOLO
IDS_MODEL_PATH = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\yolo11_container\YOLO_IDs\ID_YOLO_container\weights\best.pt"
CHAR_MODEL_PATH = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\yolo11_container\YOLO_Characters\Character_YOLO_container_finetune_extra_large_phase2\weights\best.pt"

# EasyOCR: укажи папку, где локально лежат модели
OCR_MODEL_PATH = r"C:\Users\User\.EasyOCR"

# ================================
# Инициализация моделей
# ================================
print("🖥️ Ejecutando en CPU (GPU deshabilitada)")

ids_model = YOLO(IDS_MODEL_PATH)
char_model = YOLO(CHAR_MODEL_PATH)

ocr_model = easyocr.Reader(
    ['en','es'],
    gpu=False,
    model_storage_directory=OCR_MODEL_PATH,
    download_enabled=False  # отключаем автоматическое скачивание
)

# ================================
# Regex и нормализация
# ================================
rules = {
    "code-container": {"attribute": "code-container", "regex": r"^[A-Z]{4}\d{7}$"},
    "cn-11": {"attribute": "cn-11", "regex": r"^[A-Z]{4}\d{7}$"},
    "cn-4": {"attribute": "cn-4", "regex": r"^[A-Z]{4}$"},
    "cn-7": {"attribute": "cn-7", "regex": r"^\d{7}$"},
    "iso-type": {"attribute": "iso-type", "regex": r"^\d{2}[A-Z][A-Z0-9]$"}
}

def normalize_code(value: str, key: str) -> str:
    if not value:
        return value
    if key in ["cn-11", "cn-4", "code-container"]:
        prefix = value[:4].replace("0", "O").replace("1", "I").replace("5", "S") \
                          .replace("2", "Z").replace("8", "B").replace("6", "G") \
                          .replace("4", "A").replace("7", "T")
        rest = value[4:]
        return prefix + rest
    return value

def parse_detecciones(detecciones, rules):
    parsed = {}
    for key, value in detecciones.items():
        if key in rules:
            if isinstance(value, dict) and "text" in value:
                value_to_validate = value["text"]
            else:
                value_to_validate = value
            attr = rules[key]["attribute"]
            pattern = rules[key]["regex"]
            value_norm = normalize_code(value_to_validate, key)
            match = bool(re.match(pattern, value_norm))
            parsed[attr] = {"raw": value_to_validate, "normalized": value_norm, "valid": "✔️" if match else "❌"}
    return parsed

def calculate_check_digit(container_code: str):
    if not container_code or len(container_code) != 11:
        return None
    if not container_code[:4].isalpha():
        return None
    code_10 = container_code[:10]
    letter_values = { 'A':10,'B':12,'C':13,'D':14,'E':15,'F':16,'G':17,'H':18,'I':19,'J':20,
                     'K':21,'L':23,'M':24,'N':25,'O':26,'P':27,'Q':28,'R':29,'S':30,'T':31,
                     'U':32,'V':34,'W':35,'X':36,'Y':37,'Z':38 }
    values = []
    for char in code_10:
        values.append(letter_values[char.upper()] if char.isalpha() else int(char))
    total = sum(val * (2 ** i) for i, val in enumerate(values))
    check_digit = total % 11
    if check_digit == 10: check_digit = 0
    last_digit = container_code[10]
    if last_digit.isdigit() and int(last_digit) == check_digit:
        return container_code
    return None

# ================================
# Глобальный трек кодов
# ================================
codigos_detectados = {}
next_id = 1

# ================================
# Функция обработки одного кадра
# ================================
def predict_frame(frame_bgr):
    global codigos_detectados, next_id

    image_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    # Тут используем ту же функцию predict (YOLO char + OCR)
    # Для краткости используем упрощённый вариант:
    # Возвращает annotated_frame (BGR) и salida_json
    annotated_frame, salida_json = predict(image_pil)  # твоя функция predict

    # Получаем финальный код
    codigo_final_id = salida_json.get("codigo_final_id")
    track_id = salida_json.get("codigo_final_track")
    codigo_conf = salida_json.get("codigo_final_conf", 0.0)

    if codigo_final_id:
        existing_by_track = {k:v for k,v in codigos_detectados.items() if v['track_id']==track_id}
        if track_id is not None and existing_by_track:
            existing = list(existing_by_track.values())[0]
            if codigo_conf > existing["conf"]:
                existing["code"] = codigo_final_id
                existing["conf"] = codigo_conf
                print(f"🔄 Track {track_id} updated with conf {codigo_conf:.2f} (Code={codigo_final_id})")
        else:
            if not any(v['code']==codigo_final_id for v in codigos_detectados.values()):
                codigos_detectados[next_id] = {
                    "code": codigo_final_id,
                    "track_id": track_id,
                    "conf": codigo_conf
                }
                print(f"✔️ New code detected: {codigo_final_id} | Track={track_id} | Conf={codigo_conf:.2f}")
                next_id += 1

    # Overlay на изображение
    y_offset = 70
    for info in codigos_detectados.values():
        track_display = info['track_id'] if info['track_id'] is not None else "None"
        display_text = f"{info['code']} | Track={track_display} | Conf={info['conf']:.2f}"
        cv2.putText(
            annotated_frame, display_text, (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0,255,0), 3, cv2.LINE_AA
        )
        y_offset += 50

    return annotated_frame, salida_json

# ================================
# Обработка видео
# ================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    output_name = f"label_{video_name}"
    output_path = os.path.join(video_dir, output_name)

    frame_id = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    print(f"📀 Saving annotated video to: {output_path}")

    ultimo_frame_annotated = None
    ultimo_salida_json = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_id % SKIP_FRAMES == 0:
            small_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            annotated_small, salida_json = predict_frame(small_frame)
            annotated_frame = cv2.resize(annotated_small, (frame.shape[1], frame.shape[0]))
            ultimo_frame_annotated = annotated_frame.copy()
            ultimo_salida_json = salida_json
        else:
            annotated_frame = ultimo_frame_annotated.copy()
            salida_json = ultimo_salida_json

        out.write(annotated_frame)
        frame_id += 1

    cap.release()
    out.release()
    print("✅ Video processing finished")
    return output_path

# ================================
# Обработка одного изображения
# ================================
def process_image(image_path):
    frame_bgr = cv2.imread(image_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    annotated_frame, salida_json = predict_frame(frame_bgr)
    out_path = image_path.replace(".jpg","_label.jpg").replace(".png","_label.png")
    cv2.imwrite(out_path, annotated_frame)
    # Сохраняем JSON
    with open(out_path.replace(".jpg",".json").replace(".png",".json"), "w", encoding="utf-8") as f:
        json.dump(salida_json, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved annotated image: {out_path}")
    return annotated_frame, salida_json

# ================================
# Пример использования
# ================================
if __name__ == "__main__":
    # 1️⃣ Видео
    video_path = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\videos\label_09158888.mp4"
    process_video(video_path)

    # 2️⃣ Изображение
    img_path = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\images\container1.jpg"
    process_image(img_path)
