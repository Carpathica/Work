import cv2
import numpy as np

REFERENCE_ANGLE = -45  # Ожидаемый угол (горизонтальные линии)
ANGLE_TOLERANCE = 20  # Допустимое отклонение

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Lower H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 38, 180, nothing)
cv2.createTrackbar("Lower S", "Trackbars", 64, 255, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 36, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

def extract_red_channel(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")
    
    lower_red1 = np.array([lower_h, lower_s, lower_v])
    upper_red1 = np.array([upper_h,  upper_s, upper_v])
    lower_red2 = np.array([100, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def preprocess(mask):
    """Морфологическая обработка"""
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def skeletonize(img):
    """Скелетизация"""
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9))
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

image = cv2.imread("КСП со следом далеко/5334796642802595518.jpg")
while True:
    red_mask = extract_red_channel(image)
    preprocessed_mask = preprocess(red_mask)
    skeleton = skeletonize(preprocessed_mask)
    colored_skeleton = np.zeros_like(image)
    colored_skeleton = color_skeleton_segments(skeleton, colored_skeleton)
    
    
    red_mask_rs = cv2.resize(red_mask,(720,720))
    sklt_rs = cv2.resize(skeleton,(720,720))
    sklt_col_rs = cv2.resize(colored_skeleton,(720,720))

    cv2.imshow("Red Mask", red_mask_rs)
    cv2.imshow("Skeleton", sklt_rs)
    cv2.imshow("Colored Skeleton", sklt_col_rs)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()