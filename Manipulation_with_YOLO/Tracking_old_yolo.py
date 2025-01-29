import cv2
import torch
import time

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='results/yolov5n/weights/best.pt')  # Укажите путь к вашей модели

# Загрузка видео
video_path = 'Test_Video/4.mp4'
cap = cv2.VideoCapture(video_path)

# Проверьте, удалось ли открыть видео
if not cap.isOpened():
    print(f"Не удалось открыть видео: {video_path}")
    exit()

# Настройка параметров для сохранения видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('results_video/Yolo5n_50e/yolo5n_50e_4.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Инициализация переменных для расчета времени
frame_times = []  # Список для хранения времени обработки каждого кадра
frame_count = 0  # Счетчик кадров

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Закончились кадры

    # Засекаем время начала обработки
    start_time = time.time()

    # Выполняем детекцию на текущем кадре
    results = model(frame)

    # Извлекаем и рисуем результаты на кадре
    annotated_frame = results.render()[0]  # Визуализация на изображении

    # Засекаем время окончания обработки
    end_time = time.time()
    frame_time = end_time - start_time
    frame_times.append(frame_time)
    frame_count += 1

    # Выводим время обработки текущего кадра
    print(f"Кадр {frame_count}: время обработки = {frame_time:.4f} сек")

    # Сохранение обработанного кадра
    out.write(annotated_frame)

    # Отображение результата
    cv2.imshow('YOLOv5 Detection', annotated_frame)

    # Нажмите 'q', чтобы выйти
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

# Расчет среднего времени обработки
average_time = 1/(sum(frame_times) / len(frame_times))
print(f"Среднее время обработки кадра: {average_time:.4f} сек")
print(f"Общее количество обработанных кадров: {frame_count}")
