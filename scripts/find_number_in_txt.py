import os
import sys
from pathlib import Path

def search_number_in_files(folder_path, search_number="29"):
    """
    Ищет указанное число в текстовых файлах в заданной папке
    
    Args:
        folder_path: Путь к папке для поиска
        search_number: Число для поиска (по умолчанию "29")
    """
    try:
        # Преобразуем путь в объект Path
        folder = Path(folder_path)
        
        # Проверяем, существует ли папка
        if not folder.exists():
            print(f"Ошибка: Папка '{folder_path}' не существует.")
            return
        
        if not folder.is_dir():
            print(f"Ошибка: '{folder_path}' не является папкой.")
            return
        
        print(f"Поиск числа '{search_number}' в папке: {folder.absolute()}")
        print("-" * 60)
        
        # Счетчики
        files_found = 0
        total_matches = 0
        
        # Проходим по всем файлам в папке и подпапках
        for file_path in folder.rglob("*"):
            # Пропускаем папки
            if file_path.is_dir():
                continue
            
            # Проверяем, является ли файл текстовым (по расширению)
            text_extensions = {'.txt', '.log', '.csv', '.json', '.xml', '.yaml', '.yml', 
                              '.md', '.ini', '.cfg', '.conf', '.py', '.js', '.html', '.css'}
            
            # Можно также проверять все файлы, а не только по расширению
            # Для этого уберите следующую строку и проверку расширения
            if file_path.suffix.lower() not in text_extensions:
                continue
            
            try:
                # Открываем файл в правильной кодировке
                with open(file_path, 'r', encoding='utf-8') as file:
                    line_number = 0
                    file_matches = 0
                    
                    for line in file:
                        line_number += 1
                        
                        # Ищем число в строке
                        if search_number in line:
                            file_matches += 1
                            total_matches += 1
                            
                            # Выводим информацию о найденном совпадении
                            if file_matches == 1:  # Первое совпадение в файле
                                print(f"\nФайл: {file_path.relative_to(folder)}")
                                print(f"Полный путь: {file_path}")
                            
                            # Убираем лишние пробелы для красивого вывода
                            cleaned_line = line.rstrip('\n').replace('\t', ' ')
                            # Обрезаем слишком длинные строки
                            if len(cleaned_line) > 150:
                                cleaned_line = cleaned_line[:147] + "..."
                            
                            print(f"  Строка {line_number}: {cleaned_line}")
                
                if file_matches > 0:
                    files_found += 1
                    print(f"  Всего совпадений в файле: {file_matches}")
                    
            except UnicodeDecodeError:
                # Пропускаем бинарные файлы
                continue
            except PermissionError:
                print(f"Нет доступа к файлу: {file_path}")
                continue
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")
                continue
        
        print("-" * 60)
        print(f"\nРезультаты поиска:")
        print(f"Найдено файлов с совпадениями: {files_found}")
        print(f"Всего совпадений: {total_matches}")
        
        if files_found == 0:
            print(f"Число '{search_number}' не найдено в текстовых файлах.")
            
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def main():
    """Основная функция с обработкой аргументов командной строки"""
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        search_number = sys.argv[2] if len(sys.argv) > 2 else "29"
    else:
        # Если аргументы не переданы, запрашиваем у пользователя
        folder_path = input("Введите путь к папке для поиска: ").strip()
        search_number = input("Введите число для поиска (по умолчанию 29): ").strip()
        
        if not search_number:
            search_number = "29"
    
    # Запускаем поиск
    search_number_in_files(folder_path, search_number)

if __name__ == "__main__":
    main()