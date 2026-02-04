import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

REFERENCE_ANGLE = -75  # Ожидаемый угол (горизонтальные линии)
ANGLE_TOLERANCE = 10  # Допустимое отклонение
MAX_CONTOUR_SIZE = 450  # Максимальный размер участка


def nothing(x):
    pass


cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 11, 180, nothing)
cv2.createTrackbar("Lower S", "Trackbars", 64, 255, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 19, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 73, 255, nothing)


def extract_red_channel(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")

    lower_red1 = np.array([lower_h, lower_s, lower_v])
    upper_red1 = np.array([upper_h, upper_s, upper_v])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Закрытие небольших разрывов
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
        print("Нет углов для построения графика")
        return
    angles = abs(np.radians([float(a) for a in angles]))
    x, y = [0], [0]
    base = 10.0
    for angle in angles:
        height = base * np.tan(angle / 2)
        x.extend([x[-1] + base / 2, x[-1] + base])
        y.extend([height, 0])
    if len(x) < 2 or len(y) < 2:
        print("Недостаточно точек для интерполяции")
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


image = cv2.imread("no_step\\5334796642802595539.jpg")
while True:
    red_mask = extract_red_channel(image)
    output = image.copy()
    result, detected_angles = detect_lines(red_mask, output)

    red_mask_rs = cv2.resize(red_mask, (720, 720))
    result_rs = cv2.resize(result, (720, 720))

    cv2.imshow("Red Mask", red_mask_rs)
    cv2.imshow("Detected Lines", result_rs)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
sawtooth_plot(detected_angles, "sawtooth_plot.png")
