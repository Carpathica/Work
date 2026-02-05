import os

DATASET_ROOT = r"/path/to/dataset"  # <-- укажи путь к датасету
SPLITS = ["train", "val", "test"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_files_without_ext(folder, valid_exts):
    files = set()
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            name, ext = os.path.splitext(f)
            if ext.lower() in valid_exts:
                files.add(name)
    return files


def check_split(split):
    print(f"\n=== Проверка сплита: {split} ===")

    images_dir = os.path.join(DATASET_ROOT, split, "images")
    labels_dir = os.path.join(DATASET_ROOT, split, "labels")

    if not os.path.isdir(images_dir):
        print(f"[WARN] Нет папки images: {images_dir}")
        return
    if not os.path.isdir(labels_dir):
        print(f"[WARN] Нет папки labels: {labels_dir}")
        return

    image_files = get_files_without_ext(images_dir, IMAGE_EXTS)
    label_files = get_files_without_ext(labels_dir, {".txt"})

    missing_labels = image_files - label_files
    orphan_labels = label_files - image_files

    if missing_labels:
        print(f"\n❌ Картинки без label ({len(missing_labels)}):")
        for name in sorted(missing_labels):
            print(f"  {name}")
    else:
        print("✔ У всех изображений есть label")

    if orphan_labels:
        print(f"\n⚠ Label без изображения ({len(orphan_labels)}):")
        for name in sorted(orphan_labels):
            print(f"  {name}.txt")
    else:
        print("✔ Нет лишних label-файлов")


def main():
    print("Запуск проверки соответствия изображений и разметки...")
    for split in SPLITS:
        check_split(split)
    print("\nГотово ✅")


if __name__ == "__main__":
    main()
