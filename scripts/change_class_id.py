import os

# Папка с label-файлами YOLO
LABELS_DIR = r"/path/to/your/labels"  # <-- замени на свой путь

# Какие классы и как менять
def remap_class(class_id: int) -> int:
    if class_id in [30, 31, 32, 33]:
        return class_id + 1
    elif class_id == 35:
        return 30
    else:
        return class_id


def process_label_file(file_path: str):
    new_lines = []
    changed = False

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        class_id = int(parts[0])
        new_class_id = remap_class(class_id)

        if new_class_id != class_id:
            changed = True

        parts[0] = str(new_class_id)
        new_lines.append(" ".join(parts))

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[UPDATED] {file_path}")
    else:
        print(f"[OK] {file_path}")


def main():
    for root, _, files in os.walk(LABELS_DIR):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                process_label_file(file_path)


if __name__ == "__main__":
    main()
