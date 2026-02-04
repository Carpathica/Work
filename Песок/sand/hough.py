import cv2
import numpy as np

REFERENCE_ANGLE = -80  # Ожидаемый угол (горизонтальные линии)
ANGLE_TOLERANCE = 30  # Допустимое отклонение

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("Brightness Threshold", "Trackbars", 200, 255, nothing)

def extract_laser_by_brightness(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0]  # Канал яркости
    threshold_value = cv2.getTrackbarPos("Brightness Threshold", "Trackbars")
    _, mask = cv2.threshold(l_channel, threshold_value, 255, cv2.THRESH_BINARY)  # Выделяем яркие области
    return mask

def preprocess(mask):
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def detect_laser_line(mask, output):
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return output

def process_image(image_path):
    image = cv2.imread(image_path)
    while True:
        bright_mask = extract_laser_by_brightness(image)
        preprocessed_mask = preprocess(bright_mask)
        detected_lines = detect_laser_line(preprocessed_mask, image.copy())
        
        cv2.imshow("Original", image)
        cv2.imshow("Mask", bright_mask)
        cv2.imshow("Detected Lines", detected_lines)
        
        key = cv2.waitKey(1)
        if key == 27:  # ESC для выхода
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'КСП со следом близко/5334796642802595526.jpg'  # Укажите путь к изображению
    process_image(image_path)