import cv2
import time
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('results/yolo11m/train/weights/best.pt')  # Укажите путь к вашей модели YOLOv8

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
out = cv2.VideoWriter('results_video/Yolo11m_50e/yolo11m_50e_4.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

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

    # Визуализация результатов на кадре
    annotated_frame = results[0].plot()

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
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Нажмите 'q', чтобы выйти
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

# Расчет среднего времени обработки
average_time = sum(frame_times) / len(frame_times)
average_fps = 1 / average_time if average_time > 0 else 0  # FPS
print(f"Среднее время обработки кадра: {average_time:.4f} сек")
print(f"Средний FPS: {average_fps:.2f}")
print(f"Общее количество обработанных кадров: {frame_count}")
