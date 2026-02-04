import cv2
import numpy as np
import os
import glob

REFERENCE_ANGLE = -75  # Ожидаемый угол (горизонтальные линии)
ANGLE_TOLERANCE = 30 # Допустимое отклонение


def extract_red_channel(img):
    """Фильтрация красного цвета"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    return mask

def preprocess(mask):
    """Морфологическая обработка"""
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def skeletonize(img):
    """Скелетизация"""
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))

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
    """Определение угла по точкам"""
    if len(points) < 100:
        return None
    
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.degrees(np.arctan2(vy, vx))
    
    return angle

def color_skeleton_segments(skel, output):
    """Определение углов и окрашивание пикселей напрямую"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel)

    for i in range(1, num_labels):  # Пропускаем фон (label 0)
        mask = (labels == i).astype(np.uint8) * 255
        y, x = np.where(mask > 0)
        points = np.column_stack((x, y))

        if 100 < len(points) < 750:
            angle = calculate_angle(points)
            if angle is not None:
                if REFERENCE_ANGLE - ANGLE_TOLERANCE <= angle <= REFERENCE_ANGLE + ANGLE_TOLERANCE:
                    color = (0, 255, 0)  # Зеленый
                    print(angle)
                else:
                    color = (0, 0, 255)  # Красный

                # Окрашиваем напрямую пиксели
                for (px, py) in points:
                    output[py, px] = color

    return output

# Загружаем изображение
image = cv2.imread("КСП со следом близко/5334796642802595530.jpg")

# Извлекаем красные линии
red_mask = extract_red_channel(image)

# Предобработка
preprocessed_mask = preprocess(red_mask)

# Скелетизация
skeleton = skeletonize(preprocessed_mask)

# Цветной результат
colored_skeleton = np.zeros_like(image)

# Окрашиваем сегменты
colored_skeleton = color_skeleton_segments(skeleton, colored_skeleton)

# Отображение результатов
cv2.imshow("Skeleton", skeleton)

cv2.imshow("Colored Skeleton", colored_skeleton)
cv2.waitKey(0)
cv2.destroyAllWindows()
