import os

def process_txt_files(root_folder):
    # Проход по всем папкам и файлам
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(foldername, filename)
                # Открываем и обрабатываем файл
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                # Меняем первую цифру на 0 в каждой строке
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if parts:
                        parts[0] = '0'  # Меняем первый элемент (класс)
                        new_lines.append(' '.join(parts) + '\n')
                
                # Записываем изменения обратно в файл
                with open(file_path, 'w') as file:
                    file.writelines(new_lines)

    print("Обработка завершена!")

# Укажите путь к корневой папке с файлами
root_folder = "/home/andreevaleksandr/Documents/YOLO/Drone_dataset_main"
process_txt_files(root_folder)
