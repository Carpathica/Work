import cv2
import yaml
from pathlib import Path
import random

IMAGES_DIR = Path("C:\\Users\\User\\Desktop\\Projects\\Work\\scripts\\test\\images")
LABELS_DIR = Path("C:\\Users\\User\\Desktop\\Projects\\Work\\scripts\\test\\labels")
YAML_PATH = Path("C:\\Users\\User\\Desktop\\Projects\\Work\\scripts\\data.yaml")

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
WINDOW_NAME = "YOLO Viewer"

# ---------- Загрузка имён классов ----------
with open(YAML_PATH, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

NAMES = data.get("names", [])
print(f"[INFO] Загружено классов: {len(NAMES)}")


def get_class_color(class_id):
    random.seed(class_id)
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )


def draw_yolo_boxes(image, label_path):
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, bw, bh = parts
            class_id = int(class_id)
            x_center, y_center, bw, bh = map(float, (x_center, y_center, bw, bh))

            x_center *= w
            y_center *= h
            bw *= w
            bh *= h

            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)

            color = get_class_color(class_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            if 0 <= class_id < len(NAMES):
                class_name = NAMES[class_id]
            else:
                class_name = f"id {class_id}"

            label = f"{class_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1_text = max(y1 - th - 6, 0)

            cv2.rectangle(image, (x1, y1_text), (x1 + tw, y1_text + th + 6), color, -1)
            cv2.putText(image, label, (x1, y1_text + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image


# ---------- Подготовка ----------
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

image_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])

if not image_paths:
    print("В папке images нет изображений")
    exit()

index = 0

# ---------- Основной цикл ----------
while True:
    img_path = image_paths[index]
    label_path = LABELS_DIR / (img_path.stem + ".txt")

    print(f"\n[IMAGE] {img_path.name}")  # <-- теперь имя всегда в консоли

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"Ошибка загрузки {img_path.name}")
        break

    if label_path.exists():
        image = draw_yolo_boxes(image, label_path)
    else:
        cv2.putText(image, "NO LABEL", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    info = f"{index+1}/{len(image_paths)} : {img_path.name}"
    cv2.putText(image, info, (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, image)
    cv2.setWindowTitle(WINDOW_NAME, img_path.name)

    key = cv2.waitKeyEx(0)

    if key == 27:  # ESC
        break

    # Следующее изображение
    elif key in [ord('d'), ord('D'), 2555904]:
        index = (index + 1) % len(image_paths)

    # Предыдущее изображение
    elif key in [ord('a'), ord('A'), 2424832]:
        index = (index - 1) % len(image_paths)

cv2.destroyAllWindows()
