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

IDS_MODEL_PATH = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\yolo11_container\YOLO_IDs\ID_YOLO_container\weights\best.pt"
CHAR_MODEL_PATH = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\yolo11_container\YOLO_Characters\Character_YOLO_container_finetune_extra_large_phase2\weights\best.pt"
OCR_MODEL_PATH = r"C:\Users\User\.EasyOCR"

print("🖥️ Ejecutando en CPU")

ids_model = YOLO(IDS_MODEL_PATH)
char_model = YOLO(CHAR_MODEL_PATH)

ocr_model = easyocr.Reader(
    ['en', 'es'],
    gpu=False,
    model_storage_directory=OCR_MODEL_PATH,
    download_enabled=False
)

# ================================
# ISO 6346 check-digit
# ================================
def calculate_check_digit(container_code: str):
    if not container_code or len(container_code) != 11:
        return False
    if not container_code[:4].isalpha():
        return False

    letter_values = {
        'A':10,'B':12,'C':13,'D':14,'E':15,'F':16,'G':17,'H':18,'I':19,'J':20,
        'K':21,'L':23,'M':24,'N':25,'O':26,'P':27,'Q':28,'R':29,'S':30,'T':31,
        'U':32,'V':34,'W':35,'X':36,'Y':37,'Z':38
    }

    code_10 = container_code[:10]
    values = []

    for char in code_10:
        if char.isalpha():
            values.append(letter_values.get(char.upper(), 0))
        else:
            values.append(int(char))

    total = sum(val * (2 ** i) for i, val in enumerate(values))
    check_digit = total % 11
    if check_digit == 10:
        check_digit = 0

    return container_code[10].isdigit() and int(container_code[10]) == check_digit


# ================================
# Класс трекера (без global)
# ================================
class ContainerTracker:
    def __init__(self):
        self.codes = {}
        self.next_id = 1

    def update(self, code, track_id, conf):
        if not code:
            return

        # Бонус за валидный ISO
        if calculate_check_digit(code):
            conf += 0.05

        existing = None
        for v in self.codes.values():
            if v["track_id"] == track_id:
                existing = v
                break

        if existing:
            if conf > existing["conf"]:
                existing["code"] = code
                existing["conf"] = conf
        else:
            if not any(v["code"] == code for v in self.codes.values()):
                self.codes[self.next_id] = {
                    "code": code,
                    "track_id": track_id,
                    "conf": conf
                }
                print(f"✔️ New code: {code} | Track={track_id} | Conf={conf:.2f}")
                self.next_id += 1


tracker = ContainerTracker()


# ================================
# Обработка одного кадра
# ================================
def predict_frame(original_frame_bgr):

    # уменьшаем только для инференса
    small_frame = cv2.resize(
        original_frame_bgr,
        (0, 0),
        fx=SCALE_FACTOR,
        fy=SCALE_FACTOR
    )

    image_pil = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))

    # твоя функция
    annotated_small, salida_json = predict(image_pil)

    codigo_final_id = salida_json.get("codigo_final_id")
    track_id = salida_json.get("codigo_final_track")
    codigo_conf = salida_json.get("codigo_final_conf", 0.0)

    tracker.update(codigo_final_id, track_id, codigo_conf)

    # Рисуем на оригинальном кадре
    annotated_frame = original_frame_bgr.copy()

    y_offset = 70
    for info in tracker.codes.values():
        text = f"{info['code']} | Track={info['track_id']} | Conf={info['conf']:.2f}"
        cv2.putText(
            annotated_frame,
            text,
            (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )
        y_offset += 50

    return annotated_frame, salida_json

def predict(image_pil):

    frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    salida_json = {
        "codigo_final_id": None,
        "codigo_final_track": None,
        "codigo_final_conf": 0.0
    }

    # ================================
    # 1️⃣ Детекция контейнера (ID YOLO)
    # ================================
    results_ids = ids_model(frame_bgr, verbose=False)[0]

    if results_ids.boxes is None or len(results_ids.boxes) == 0:
        return frame_bgr, salida_json

    best_box = None
    best_conf = 0

    for box in results_ids.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is None:
        return frame_bgr, salida_json

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    track_id = None

    if hasattr(best_box, "id") and best_box.id is not None:
        track_id = int(best_box.id[0])

    roi = frame_bgr[y1:y2, x1:x2]

    # Рисуем bbox контейнера
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ================================
    # 2️⃣ Детекция символов
    # ================================
    results_chars = char_model(roi, verbose=False)[0]

    if results_chars.boxes is None or len(results_chars.boxes) == 0:
        return frame_bgr, salida_json

    characters = []

    for box in results_chars.boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = char_model.names[cls_id]

        cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
        characters.append((cx1, label, conf))

        # рисуем bbox символа
        cv2.rectangle(
            frame_bgr,
            (x1 + cx1, y1 + cy1),
            (x1 + cx2, y1 + cy2),
            (255, 0, 0),
            1
        )

    # ================================
    # 3️⃣ Сортировка слева направо
    # ================================
    characters = sorted(characters, key=lambda x: x[0])

    text = ""
    total_conf = 0

    for _, label, conf in characters:
        text += label
        total_conf += conf

    avg_conf = total_conf / len(characters)

    # ================================
    # 4️⃣ Финальный результат
    # ================================
    salida_json["codigo_final_id"] = text
    salida_json["codigo_final_track"] = track_id
    salida_json["codigo_final_conf"] = avg_conf

    # Отображаем текст
    cv2.putText(
        frame_bgr,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return frame_bgr, salida_json

# ================================
# Видео
# ================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    output_path = os.path.join(video_dir, f"label_{video_name}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    print(f"📀 Saving to: {output_path}")

    frame_id = 0
    last_annotated = None
    last_json = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % SKIP_FRAMES == 0:
            annotated_frame, salida_json = predict_frame(frame)
            last_annotated = annotated_frame.copy()
            last_json = salida_json
        else:
            annotated_frame = last_annotated.copy()
            salida_json = last_json

        out.write(annotated_frame)
        frame_id += 1

    cap.release()
    out.release()

    print("✅ Video finished")
    return output_path


# ================================
# Изображение
# ================================
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(image_path)

    annotated_frame, salida_json = predict_frame(frame)

    out_path = image_path.replace(".jpg", "_label.jpg").replace(".png", "_label.png")

    cv2.imwrite(out_path, annotated_frame)

    with open(out_path.replace(".jpg", ".json").replace(".png", ".json"),
              "w",
              encoding="utf-8") as f:
        json.dump(salida_json, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved: {out_path}")
    return annotated_frame, salida_json


# ================================
# Пример использования
# ================================
if __name__ == "__main__":

    video_path = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\videos\label_09158888.mp4"
    process_video(video_path)

    img_path = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\images\container1.jpg"
    process_image(img_path)
