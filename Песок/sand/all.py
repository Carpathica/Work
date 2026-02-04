import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

REFERENCE_ANGLE = -90 # Ожидаемый угол (горизонтальные линии)
ANGLE_TOLERANCE = 1 # Допустимое отклонение

def extract_red_channel(img):
    """Фильтрация красного цвета"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 45])
    upper_red1 = np.array([30, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    return mask

def preprocess(mask):
    """Морфологическая обработка"""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def skeletonize(img):
    """Скелетизация"""
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
    """Определение угла по точкам"""
    if  len(points) < 200:
        return None
    
    [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.degrees(np.arctan2(vy, vx))
    
    return angle

def color_skeleton_segments(skel, output):
    """Определение углов и окрашивание пикселей напрямую"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel)
    angles = []

    for i in range(1, num_labels):  # Пропускаем фон (label 0)
        mask = (labels == i).astype(np.uint8) * 255
        y, x = np.where(mask > 0)
        points = np.column_stack((x, y))

        if 220 < len(points) < 700:
            angle = calculate_angle(points)
            if angle is not None:
                if angle >= REFERENCE_ANGLE + ANGLE_TOLERANCE:
                    color = (0, 255, 0)  # Зеленый
                    angles.append(angle)
                else:
                    color = (0, 0, 255)  # Красный

                # Окрашиваем напрямую пиксели
                for (px, py) in points:
                    output[py, px] = color

    return output, angles

def sawtooth_plot(angles):
    if not angles:
        print("Нет углов для построения графика")
        return

    # Преобразуем список массивов в плоский список чисел
    angles = [float(a) for a in angles]  
    angles = abs(np.radians(angles))  # Переводим углы в радианы

    x = [0]  # Начальная точка по X
    y = [0]  # Начальная точка по Y
    base = 30.0  # Основание треугольников
    
    for angle in angles:
        height = base * np.tan(angle / 2)  # Вычисляем высоту треугольника
        x.extend([x[-1] + base / 2, x[-1] + base])
        y.extend([height, 0])

    # Проверяем, есть ли хотя бы 2 точки для интерполяции
    if len(x) < 2 or len(y) < 2:
        print("Недостаточно точек для интерполяции")
        return

    # Интерполяция для сглаживания
    cs = CubicSpline(x, y, bc_type='natural')
    x_smooth = np.linspace(min(x), max(x), 300)
    y_smooth = cs(x_smooth)

    plt.plot(x_smooth, y_smooth, linestyle='-', label='Восстановленная поверхность')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()



# Загружаем изображение
image = cv2.imread("no_step/5334796642802595539.jpg")

# Извлекаем красные линии
red_mask = extract_red_channel(image)

# Предобработка
preprocessed_mask = preprocess(red_mask)

# Скелетизация
skeleton = skeletonize(preprocessed_mask)

# Цветной результат
colored_skeleton = np.zeros_like(image)

# Окрашиваем сегменты
colored_skeleton,angles = color_skeleton_segments(skeleton, colored_skeleton)

# Отображение результатов
sklt_rs = cv2.resize(skeleton,(720,720))
sklt_col_rs = cv2.resize(colored_skeleton,(720,720))
cv2.imshow("Skeleton", sklt_rs)
cv2.imshow("Colored Skeleton", sklt_col_rs)
cv2.waitKey(0)
cv2.destroyAllWindows()
sawtooth_plot(angles)
