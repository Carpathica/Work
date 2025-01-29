import os
import zipfile

def extract_zip_files_from_directories(source_dirs, target_dir):
    # Создаем целевую папку, если её нет
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for source_dir in source_dirs:
        # Рекурсивно обходим каждую исходную папку и её подпапки
        for root, _, files in os.walk(source_dir):
            for file in files:
                # Ищем файлы с именем, содержащим 'txt.zip'
                if file.lower().endswith('txt.zip'):
                    zip_path = os.path.join(root, file)
                    extract_to = os.path.join(target_dir, os.path.splitext(file)[0])

                    # Создаем папку для извлечения архива
                    if not os.path.exists(extract_to):
                        os.makedirs(extract_to)

                    # Извлекаем архив
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                        print(f"Extracted {zip_path} to {extract_to}")

# Пример использования
source_dirs = [
    '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_from_Ivan/Anafi-Extended (VisioDECT Dataset)/labels/evening',
    '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_from_Ivan/Anafi-Extended (VisioDECT Dataset)/labels/sunny',
    '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_from_Ivan/DJIFPV (VisioDECT Dataset)',
    '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_from_Ivan/DJIPhantom (VisioDECT Dataset)',
    '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_from_Ivan/EFT-E410S (VisioDECT Dataset)',
    '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_from_Ivan/Mavic_Air (VisioDECT Dataset)/images'

]

target_dir = '/home/andreevaleksandr/Documents/YOLO/all_in_one_txt'

extract_zip_files_from_directories(source_dirs, target_dir)
