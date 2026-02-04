import cv2
import numpy as np

REFERENCE_ANGLE = -75  # Ожидаемый угол (горизонтальные линии)
ANGLE_TOLERANCE = 30  # Допустимое отклонение

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower H1", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("Upper H1", "Trackbars", 10, 180, nothing)
cv2.createTrackbar("Lower H2", "Trackbars", 170, 180, nothing)
cv2.createTrackbar("Upper H2", "Trackbars", 180, 180, nothing)

def extract_red_channel(img):
    """Фильтрация красного цвета"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lh1 = cv2.getTrackbarPos("Lower H1", "Trackbars")
    uh1 = cv2.getTrackbarPos("Upper H1", "Trackbars")
    lh2 = cv2.getTrackbarPos("Lower H2", "Trackbars")
    uh2 = cv2.getTrackbarPos("Upper H2", "Trackbars")
    
    lower_red1 = np.array([lh1, 70, 50])
    upper_red1 = np.array([uh1, 255, 255])
    lower_red2 = np.array([lh2, 70, 50])
    upper_red2 = np.array([uh2, 255, 255])
    
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
    for i in range(1, num_labels):
        mask = (labels == i).astype(np.uint8) * 255
        y, x = np.where(mask > 0)
        points = np.column_stack((x, y))
        if 100 < len(points) < 750:
            angle = calculate_angle(points)
            if angle is not None:
                color = (0, 255, 0) if (REFERENCE_ANGLE - ANGLE_TOLERANCE <= angle <= REFERENCE_ANGLE + ANGLE_TOLERANCE) else (0, 0, 255)
                for (px, py) in points:
                    output[py, px] = color
    return output

image = cv2.imread("КСП со следом близко/5334796642802595530.jpg")
while True:
    red_mask = extract_red_channel(image)
    preprocessed_mask = preprocess(red_mask)
    skeleton = skeletonize(preprocessed_mask)
    colored_skeleton = np.zeros_like(image)
    colored_skeleton = color_skeleton_segments(skeleton, colored_skeleton)
    
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Skeleton", skeleton)
    cv2.imshow("Colored Skeleton", colored_skeleton)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
