import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

REFERENCE_ANGLE = -75  # Ожидаемый угол (горизонтальные линии)
ANGLE_TOLERANCE = 10  # Допустимое отклонение
MAX_CONTOUR_SIZE = 450  # Максимальный размер участка


def extract_red_channel(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 64, 19])
    upper_red1 = np.array([11, 255, 73])
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


def sawtooth_plot(angles, output_path):
    if not angles:
        return
    angles = abs(np.radians([float(a) for a in angles]))
    x, y = [0], [0]
    base = 10.0
    for angle in angles:
        height = base * np.tan(angle / 2)
        x.extend([x[-1] + base / 2, x[-1] + base])
        y.extend([height, 0])
    if len(x) < 2 or len(y) < 2:
        return
    cs = CubicSpline(x, y, bc_type="natural")
    x_smooth = np.linspace(min(x), max(x), 300)
    y_smooth = cs(x_smooth)
    plt.plot(x_smooth, y_smooth, linestyle="-")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            red_mask = extract_red_channel(image)
            output = image.copy()
            result, detected_angles = detect_lines(red_mask, output)

            mask_path = os.path.join(output_folder, f"mask_{filename}")
            detected_path = os.path.join(output_folder, f"detected_{filename}")
            plot_path = os.path.join(output_folder, f"plot_{filename}.png")

            cv2.imwrite(mask_path, red_mask)
            cv2.imwrite(detected_path, result)
            sawtooth_plot(detected_angles, plot_path)

            print(f"Обработано: {filename}")


input_folder = "no_step"
output_folder = "without_step_output"
process_folder(input_folder, output_folder)
print("Обработка завершена. Файлы сохранены в папке output.")
