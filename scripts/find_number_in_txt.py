import os
import argparse
from pathlib import Path

def find_files_with_class(label_dir, target_class, file_ext='.txt'):
    """
    Находит все файлы разметки, содержащие указанный класс
    
    Args:
        label_dir (str/Path): Директория с файлами разметки YOLO (.txt)
        target_class (int): Номер класса для поиска (начинается с 0)
        file_ext (str): Расширение файлов разметки
    
    Returns:
        list: Список имен файлов (без расширения), содержащих целевой класс
    """
    matching_files = []
    
    # Преобразуем в Path объект для удобства
    label_path = Path(label_dir)
    
    # Проверяем существование директории
    if not label_path.exists():
        raise FileNotFoundError(f"Директория '{label_dir}' не найдена")
    
    # Получаем все файлы разметки
    label_files = list(label_path.glob(f"*{file_ext}"))
    
    if not label_files:
        print(f"Предупреждение: В директории '{label_dir}' не найдено файлов {file_ext}")
        return matching_files
    
    print(f"Найдено {len(label_files)} файлов разметки")
    print(f"Поиск класса {target_class}...")
    
    # Проходим по всем файлам разметки
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Проверяем каждую строку на наличие целевого класса
            for line in lines:
                line = line.strip()
                if line:  # Пропускаем пустые строки
                    # В YOLO формат: class x_center y_center width height
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        if int(parts[0]) == target_class:
                            # Добавляем имя файла без расширения
                            matching_files.append(label_file.stem)
                            break  # Переходим к следующему файлу
                            
        except Exception as e:
            print(f"Ошибка при чтении файла {label_file}: {e}")
            continue
    
    return matching_files

def main():
    parser = argparse.ArgumentParser(
        description='Поиск файлов разметки YOLO, содержащих указанный класс'
    )
    parser.add_argument(
        '--label-dir', 
        type=str, 
        required=True,
        help='Путь к директории с файлами разметки YOLO (.txt)'
    )
    parser.add_argument(
        '--class-id', 
        type=int, 
        required=True,
        help='ID класса для поиска (начинается с 0)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='',
        help='Путь к файлу для сохранения результатов (необязательно)'
    )
    parser.add_argument(
        '--with-ext', 
        action='store_true',
        help='Выводить имена файлов с расширениями изображений'
    )
    
    args = parser.parse_args()
    
    try:
        # Находим файлы
        matching_files = find_files_with_class(args.label_dir, args.class_id)
        
        # Выводим результаты
        print(f"\n{'='*50}")
        print(f"НАЙДЕНО ФАЙЛОВ С КЛАССОМ {args.class_id}: {len(matching_files)}")
        print('='*50)
        
        if matching_files:
            print("Список файлов:")
            for i, filename in enumerate(matching_files, 1):
                if args.with_ext:
                    # Проверяем существующие расширения изображений
                    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                    for ext in image_extensions:
                        image_path = Path(args.label_dir).parent / 'images' / f"{filename}{ext}"
                        if image_path.exists():
                            print(f"{i}. {filename}{ext}")
                            break
                    else:
                        print(f"{i}. {filename} (изображение не найдено)")
                else:
                    print(f"{i}. {filename}")
        else:
            print(f"Файлы с классом {args.class_id} не найдены")
        
        # Сохраняем в файл, если указан
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for filename in matching_files:
                    f.write(f"{filename}\n")
            print(f"\nРезультаты сохранены в: {args.output}")
            
        # Статистика
        label_path = Path(args.label_dir)
        total_files = len(list(label_path.glob("*.txt")))
        if total_files > 0:
            percentage = (len(matching_files) / total_files) * 100
            print(f"\nСтатистика: {len(matching_files)}/{total_files} файлов ({percentage:.1f}%) содержат класс {args.class_id}")
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()