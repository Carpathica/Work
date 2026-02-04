import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

REFERENCE_ANGLE = -45  # Ожидаемый угол (горизонтальные линии)
ANGLE_TOLERANCE = 20  # Допустимое отклонение
MAX_CONTOUR_SIZE = 750  # Максимальный размер участка


def extract_red_channel(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 114, 0])
    upper_red1 = np.array([29, 255, 43])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_lines(mask, output):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for contour in contours:
        if 50 < len(contour) < MAX_CONTOUR_SIZE:
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = np.degrees(np.arctan2(vy, vx))
            if (
                REFERENCE_ANGLE - ANGLE_TOLERANCE
                <= angle
                <= REFERENCE_ANGLE + ANGLE_TOLERANCE
            ):
                angles.append(angle)
                left_x = int(x - vx * 50)
                left_y = int(y - vy * 50)
                right_x = int(x + vx * 50)
                right_y = int(y + vy * 50)
                cv2.line(output, (left_x, left_y), (right_x, right_y), (0, 255, 0), 2)
    return output, angles


def generate_plot(angles, width, height):
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.clear()
    if angles:
        angles = abs(np.radians([float(a) for a in angles]))
        x, y = [0], [0]
        base = 10.0
        for angle in angles:
            height = base * np.tan(angle / 2)
            x.extend([x[-1] + base / 2, x[-1] + base])
            y.extend([height, 0])
        if len(x) >= 2 and len(y) >= 2:
            cs = CubicSpline(x, y, bc_type="natural")
            x_smooth = np.linspace(min(x), max(x), 300)
            y_smooth = cs(x_smooth)
            ax.plot(x_smooth, y_smooth, linestyle="-")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)


def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видео.")
        return

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    combined_width = width * 3
    out = cv2.VideoWriter(output_video, fourcc, fps, (combined_width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        red_mask = extract_red_channel(frame)
        output = frame.copy()
        result, detected_angles = detect_lines(red_mask, output)
        plot_img = generate_plot(detected_angles, width, height)

        combined_frame = np.hstack(
            (frame, cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR), plot_img)
        )
        out.write(combined_frame)

    cap.release()
    out.release()
    print("Обработка завершена. Видео сохранено.")


input_video = "2025-0228-Камер и лазер движутся - КСП без следа-близко.mp4"
output_video = "output_video.avi"
process_video(input_video, output_video)
