import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline

REFERENCE_ANGLE = -80  
ANGLE_TOLERANCE = 30  
SCALE_FACTOR = 0.5  # Коэффициент уменьшения масштаба

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 26, 180, nothing)
cv2.createTrackbar("Lower S", "Trackbars", 36, 255, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 183, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 24, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 180, 255, nothing)

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
    lower_red2 = np.array([170, 150, 164])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def preprocess(mask):
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def skeletonize(img):
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (21, 21))
    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

def calculate_angle(points):
    if len(points) < 100:
        return None
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    return np.degrees(np.arctan2(vy, vx))

def color_skeleton_segments(skel, output):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel)
    angles = []
    for i in range(1, num_labels):
        mask = (labels == i).astype(np.uint8) * 255
        y, x = np.where(mask > 0)
        points = np.column_stack((x, y))
        if 100 < len(points) < 750:
            angle = calculate_angle(points)
            if angle is not None:
                color = (0, 255, 0) if REFERENCE_ANGLE - ANGLE_TOLERANCE <= angle <= REFERENCE_ANGLE + ANGLE_TOLERANCE else (0, 0, 255)
                angles.append(angle)
                for (px, py) in points:
                    output[py, px] = color
    return output, angles

def plot_real_time(angles):
    plt.clf()
    if not angles:
        return
    angles = abs(np.radians([float(a) for a in angles]))
    x, y = [0], [0]
    base = 3.0
    for angle in angles:
        height = base * np.tan(angle / 2)
        x.extend([x[-1] + base / 2, x[-1] + base])
        y.extend([height, 0])
    
    if len(x) < 2 or len(y) < 2:
        return
    
    cs = CubicSpline(x, y, bc_type='natural')
    x_smooth = np.linspace(min(x), max(x), 300)
    y_smooth = cs(x_smooth)
    
    plt.plot(x_smooth, y_smooth, linestyle='-', label='Smoothed Sawtooth')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)

def process_video(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео")
        return

    frame_width = int(cap.get(3) * SCALE_FACTOR)
    frame_height = int(cap.get(4) * SCALE_FACTOR)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    plt.ion()  # Включаем интерактивный режим графика
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (frame_width, frame_height))  # Уменьшаем масштаб
        
        red_mask = extract_red_channel(frame)
        preprocessed_mask = preprocess(red_mask)
        skeleton = skeletonize(preprocessed_mask)
        colored_skeleton = np.zeros_like(frame)
        colored_skeleton, angles = color_skeleton_segments(skeleton, colored_skeleton)

        plot_real_time(angles)  # Обновление графика

        cv2.imshow("Original", frame)
        cv2.imshow("Red Mask", red_mask)
        cv2.imshow("Skeleton", colored_skeleton)

        out.write(colored_skeleton)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    plt.ioff()  # Выключаем интерактивный режим
    plt.show()
    cv2.destroyAllWindows()
    print(f"Обработано {frame_count} кадров")

if __name__ == "__main__":
    video_path = "input_video.mp4"
    output_video_path = "output_video.avi"
    process_video(video_path, output_video_path)

