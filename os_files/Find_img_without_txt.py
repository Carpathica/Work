import os

# Параметры
image_folder = '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_main/valid/images'  # Папка с изображениями
annotation_folder = '/home/andreevaleksandr/Documents/YOLO/Drone_dataset_main/valid/labels'  # Папка с разметками

# Получение всех файлов изображений и разметок
images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
annotations = [f for f in os.listdir(annotation_folder) if f.endswith('.txt')]

# Приведение к множествам без расширений для сравнения
image_names = set(os.path.splitext(image)[0] for image in images)
annotation_names = set(os.path.splitext(ann)[0] for ann in annotations)

# Найти изображения, для которых нет разметки
missing_annotations = image_names - annotation_names
if missing_annotations:
    print(f"Для следующих изображений нет разметки: {missing_annotations}")
else:
    print("Все изображения имеют разметки.")

# Найти разметки, для которых нет изображений
extra_annotations = annotation_names - image_names
if extra_annotations:
    print(f"Найдены лишние файлы разметки: {extra_annotations}")
else:
    print("Все разметки соответствуют изображениям.")
