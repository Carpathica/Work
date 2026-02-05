import os
import hashlib
from collections import defaultdict

DATASET_ROOT = r"/path/to/dataset"  # <-- замени на свой путь

SPLITS = ["train", "val", "test"]
SUBFOLDERS = ["images", "labels"]


def file_hash(path, chunk_size=8192):
    """Считаем md5 хеш файла (по содержимому)"""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_files():
    """Собираем информацию о всех файлах"""
    files_by_name = defaultdict(list)   # filename -> [(split, type, full_path)]
    files_by_hash = defaultdict(list)   # hash -> [(split, type, filename, full_path)]

    for split in SPLITS:
        for sub in SUBFOLDERS:
            folder = os.path.join(DATASET_ROOT, split, sub)
            if not os.path.isdir(folder):
                continue

            for root, _, files in os.walk(folder):
                for fname in files:
                    full_path = os.path.join(root, fname)

                    files_by_name[fname].append((split, sub, full_path))

                    try:
                        h = file_hash(full_path)
                        files_by_hash[h].append((split, sub, fname, full_path))
                    except Exception as e:
                        print(f"[ERROR HASH] {full_path}: {e}")

    return files_by_name, files_by_hash


def check_name_collisions(files_by_name):
    print("\n=== Совпадения ИМЁН файлов между сплитами ===")
    found = False

    for fname, entries in files_by_name.items():
        splits = set(e[0] for e in entries)
        if len(splits) > 1:
            found = True
            print(f"\n[NAME DUPLICATE] {fname}")
            for split, sub, path in entries:
                print(f"  - {split}/{sub}: {path}")

    if not found:
        print("✔ Совпадений по именам не найдено")


def check_hash_collisions(files_by_hash):
    print("\n=== Совпадения по СОДЕРЖИМОМУ (hash) между сплитами ===")
    found = False

    for h, entries in files_by_hash.items():
        splits = set(e[0] for e in entries)
        if len(entries) > 1 and len(splits) > 1:
            found = True
            print(f"\n[CONTENT DUPLICATE] hash={h}")
            for split, sub, fname, path in entries:
                print(f"  - {split}/{sub}: {fname} ({path})")

    if not found:
        print("✔ Совпадений по содержимому не найдено")


def main():
    print("Сканируем датасет...")
    files_by_name, files_by_hash = collect_files()

    check_name_collisions(files_by_name)
    check_hash_collisions(files_by_hash)

    print("\nГотово.")


if __name__ == "__main__":
    main()
