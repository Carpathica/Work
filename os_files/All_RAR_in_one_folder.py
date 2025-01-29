import os
import shutil

def collect_txt_rar_files_from_directories(source_dirs, target_dir):
    # Создаем целевую папку, если её нет
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for source_dir in source_dirs:
        # Рекурсивно обходим каждую исходную папку и её подпапки
        for root, _, files in os.walk(source_dir):
            for file in files:
                # Ищем файлы с названием, содержащим 'txt.rar'
                if file.lower().endswith('txt.rar'):
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_dir, file)

                    # Копируем файл в целевую папку
                    shutil.copy2(source_file, target_file)
                    print(f"Copied {source_file} to {target_file}")

# Пример использования
source_dirs = [
    '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_from_Ivan/Mavic_Enterprise (VisioDECT Dataset)/labels',
    '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_from_Ivan/Mavic_Air (VisioDECT Dataset)/labels'
]

target_dir = '/home/andreevaleksandr/Documents/YOLO/all_in_one_txt'

collect_txt_rar_files_from_directories(source_dirs, target_dir)
