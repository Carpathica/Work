import os
import json
import math
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple

# ================== НАСТРОЙКИ ==================

ROI_MODEL_PATH = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\yolo11_container\YOLO_IDs\ID_YOLO_container\weights\best.pt"
CHAR_MODEL_PATH = r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\yolo11_container\YOLO_Characters\Character_YOLO_container_finetune_large\weights\best.pt"


ROI_CONF = 0.35
ROI_IOU = 0.5

CHAR_CONF_RAW = 0.5   # низкий порог при сборе raw детекций, чтобы ничего не потерять
CHAR_IOU_RAW = 0.5    # слабая NMS при сборе, мы сами будем объединять

MIN_ROI_HEIGHT = 128   # минимальная высота ROI для детекции символов (масштабируем при необходимости)

MERGE_EPS = 0.005       # относительный порог (доля от ROI высоты) для кластеризации по центрам
MERGE_IOU_SIM = 1    # вспомогательный порог для merge (не обязателен, используется внутри)
# ================================================

# Карты похожих символов
ALPHA_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}
DIGIT_TO_ALPHA = {v: k for k, v in ALPHA_TO_DIGIT.items()}

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")

CONF_KEEP_THRESHOLD = 0.85  # порог, выше которого не корректируем символы

# ISO6346 letter values (approx; can be tuned/extended)
LETTER_VALUE = {}
val = 10
skip_vals = {11, 22, 33}
for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    while val in skip_vals:
        val += 1
    LETTER_VALUE[ch] = val
    val += 1

DEBUG = False  # поставь True чтобы видеть распечатки и debug-изображения

# ================================================
# Утилиты
# ================================================

def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / (areaA + areaB - inter + 1e-6)

def enlarge_bbox(bbox: List[int], pad: int, w_limit: int, h_limit: int) -> Tuple[int,int,int,int]:
    x1, y1, x2, y2 = bbox
    x1n = max(0, x1 - pad)
    y1n = max(0, y1 - pad)
    x2n = min(w_limit - 1, x2 + pad)
    y2n = min(h_limit - 1, y2 + pad)
    return x1n, y1n, x2n, y2n

def iso6346_check_digit(code10: str) -> int:
    total = 0
    for pos, ch in enumerate(code10[:10]):
        if ch.isalpha():
            v = LETTER_VALUE.get(ch.upper(), 0)
        elif ch.isdigit():
            v = int(ch)
        else:
            v = 0
        total += v * (1 << pos)
    remainder = total % 11
    return 0 if remainder == 10 else remainder

# ================================================
# Merge / clustering детекций (WBF-подобно)
# ================================================
def cluster_and_merge(chars: List[Dict], roi_h: int, eps_rel: float = MERGE_EPS) -> List[Dict]:
    """
    chars: list of {'bbox':[x1,y1,x2,y2], 'label':str, 'confidence':float}
    roi_h: высота ROI (для нормировки расстояний)
    eps_rel: порог расстояния относительно roi_h для объединения в кластер (обычно 0.06..0.12)
    Возвращает список merged элементов:
      {'bbox':[x1,y1,x2,y2], 'label': chosen_label, 'confidence': merged_conf, 'members': [...]}
    Правило выбора label: суммарная confidence по label в кластере -> label с max sum
    Координаты: взвешенное average по confidence
    """
    if len(chars) == 0:
        return []

    # центры
    centers = []
    for c in chars:
        x1,y1,x2,y2 = c['bbox']
        centers.append(((x1 + x2)/2.0, (y1 + y2)/2.0))

    eps_px = eps_rel * roi_h

    clusters = []  # list of lists of indices
    for i, center in enumerate(centers):
        assigned = False
        for cl in clusters:
            # distance to cluster centroid
            cl_centroid = np.mean([centers[j] for j in cl], axis=0)
            dist = math.hypot(center[0]-cl_centroid[0], center[1]-cl_centroid[1])
            if dist <= eps_px:
                cl.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    merged = []
    for cl in clusters:
        members = [chars[i] for i in cl]
        # aggregate confidences by label
        label_scores = {}
        for m in members:
            lab = m['label'].upper()
            label_scores[lab] = label_scores.get(lab, 0.0) + float(m['confidence'])

        # choose label with highest total score
        chosen_label = max(label_scores.items(), key=lambda x: x[1])[0]

        # weighted bbox average (weights = confidence)
        total_w = sum([m['confidence'] for m in members]) + 1e-9
        x1 = sum([m['bbox'][0]*m['confidence'] for m in members]) / total_w
        y1 = sum([m['bbox'][1]*m['confidence'] for m in members]) / total_w
        x2 = sum([m['bbox'][2]*m['confidence'] for m in members]) / total_w
        y2 = sum([m['bbox'][3]*m['confidence'] for m in members]) / total_w

        merged_conf = max([m['confidence'] for m in members])  # or sum, but keep max to stay normalized

        merged.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'label': chosen_label,
            'confidence': float(merged_conf),
            'members': members
        })

    return merged

# ================================================
# Sorting символов
# ================================================
def detect_orientation_from_boxes(boxes: List[Dict]) -> str:
    if len(boxes) < 2:
        return "horizontal"
    centers_x = [(b['bbox'][0]+b['bbox'][2])/2.0 for b in boxes]
    centers_y = [(b['bbox'][1]+b['bbox'][3])/2.0 for b in boxes]
    x_var = np.var(centers_x)
    y_var = np.var(centers_y)
    # безопасные коэффициенты
    if y_var > x_var * 1.5:
        return "vertical"
    if x_var > y_var * 1.5:
        return "horizontal"
    # ambiguous -> use median split by y (multi-line) but default left->right
    return "horizontal"

def sort_merged(merged: List[Dict]) -> List[Dict]:
    orientation = detect_orientation_from_boxes(merged)
    if orientation == "horizontal":
        return sorted(merged, key=lambda c: c['bbox'][0])  # left->right
    else:
        return sorted(merged, key=lambda c: c['bbox'][1])  # top->bottom (vertical)
# ================================================
# Correction символов по формату
# ================================================
def confidence_aware_correction(sorted_boxes: List[Dict], roi_format: str) -> Tuple[str, List[Dict]]:
    corrected_chars = []
    for i, m in enumerate(sorted_boxes):
        lab = m['label'].upper()
        conf = float(m['confidence'])
        expected_letter = None
        if roi_format == "cn-4":
            expected_letter = True
        elif roi_format in ("cn-11", "cn-7"):
            expected_letter = i < 4
        else:
            expected_letter = None

        corrected = lab
        if expected_letter is True and lab in DIGITS and conf < CONF_KEEP_THRESHOLD:
            corrected = DIGIT_TO_ALPHA.get(lab, lab)
        elif expected_letter is False and lab in LETTERS and conf < CONF_KEEP_THRESHOLD:
            corrected = ALPHA_TO_DIGIT.get(lab, lab)
        # if very low conf, try both maps
        elif conf < 0.35:
            if lab in ALPHA_TO_DIGIT:
                corrected = ALPHA_TO_DIGIT[lab]
            elif lab in DIGIT_TO_ALPHA:
                corrected = DIGIT_TO_ALPHA[lab]

        m['corrected_label'] = corrected
        corrected_chars.append(m)
    text = "".join([c['corrected_label'] for c in corrected_chars])
    return text, corrected_chars

# ================================================
# Main recognizer
# ================================================
class ISO6346AdvancedRecognizer:
    def __init__(self, roi_model_path=ROI_MODEL_PATH, char_model_path=CHAR_MODEL_PATH):
        self.roi_model = YOLO(roi_model_path)
        self.char_model = YOLO(char_model_path)

    def detect_rois(self, image: np.ndarray, conf=ROI_CONF, iou=ROI_IOU) -> List[Dict]:
        r = self.roi_model(image, conf=conf, iou=iou, verbose=False)
        rois = []
        for res in r:
            if getattr(res, 'boxes', None) is None or len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            for box, cf, cl in zip(boxes, confs, classes):
                rois.append({
                    'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    'confidence': float(cf),
                    'label': str(self.roi_model.names[int(cl)])
                })
        return rois

    def detect_raw_chars(self, roi_img: np.ndarray) -> List[Dict]:
        # scale ROI to ensure minimal height
        h, w = roi_img.shape[:2]
        if h < MIN_ROI_HEIGHT:
            scale = MIN_ROI_HEIGHT / float(h)
            roi_img = cv2.resize(roi_img, (int(w*scale), MIN_ROI_HEIGHT), interpolation=cv2.INTER_LINEAR)
            if DEBUG:
                print(f"Scaled ROI from h={h} to {MIN_ROI_HEIGHT}, scale={scale:.2f}")

        res = self.char_model(roi_img, conf=CHAR_CONF_RAW, iou=CHAR_IOU_RAW, verbose=False)
        chars = []
        for r in res:
            if getattr(r, 'boxes', None) is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for box, cf, cl in zip(boxes, confs, classes):
                lab = str(self.char_model.names[int(cl)]).upper()
                chars.append({
                    'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    'confidence': float(cf),
                    'label': lab
                })
        return chars, roi_img

    def process_image(self, image_path: str, out_image_path: str, out_json_path: str, pad: int = 8):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")

        H, W = img.shape[:2]
        rois = self.detect_rois(img)
        output = {'image': os.path.basename(image_path), 'rois': []}

        for i, roi in enumerate(rois):
            x1,y1,x2,y2 = roi['bbox']
            x1p, y1p, x2p, y2p = enlarge_bbox([x1,y1,x2,y2], pad, W, H)
            roi_crop = img[y1p:y2p, x1p:x2p].copy()

            raw_chars, roi_for_vis = self.detect_raw_chars(roi_crop)
            if DEBUG:
                print(f"\nROI {i} raw detections: {[ (c['label'], round(c['confidence'],3)) for c in raw_chars ]}")

            # If nothing -> continue
            if len(raw_chars) == 0:
                output['rois'].append({
                    'format': roi['label'],
                    'roi_bbox': [x1p,y1p,x2p,y2p],
                    'confidence': roi['confidence'],
                    'raw_text': '',
                    'corrected_text': '',
                    'checksum_valid': None,
                    'characters': []
                })
                continue

            # Merge via clustering (WBF-like)
            merged = cluster_and_merge(raw_chars, roi_h=(y2p-y1p), eps_rel=MERGE_EPS)
            if DEBUG:
                print("Merged:", [(m['label'], round(m['confidence'],3)) for m in merged])

            # Sort merged boxes
            merged_sorted = sort_merged(merged)

            # Correction
            corrected_text, corrected_chars = confidence_aware_correction(merged_sorted, roi['label'])

            # compute absolute bbox for each char (relative to original image)
            for c in corrected_chars:
                bx1, by1, bx2, by2 = c['bbox']
                # note: if ROI was scaled up in detect_raw_chars, the coords refer to scaled roi.
                # We used scaling only internally; since we returned scaled roi from detect_raw_chars,
                # we need to compute scale factor between roi_crop and roi_for_vis:
                h_orig = roi_crop.shape[0]
                h_used = roi_for_vis.shape[0]
                scale = h_orig / float(h_used)
                # but if we scaled up, roi_for_vis has MIN_ROI_HEIGHT; else equal
                bx1a = x1p + int(round(bx1 * scale))
                by1a = y1p + int(round(by1 * scale))
                bx2a = x1p + int(round(bx2 * scale))
                by2a = y1p + int(round(by2 * scale))
                c['bbox_abs'] = [bx1a, by1a, bx2a, by2a]

            # Checksum if cn-11
            checksum_valid = None
            check_digit = None
            if roi['label'] == 'cn-11' and len(corrected_text) >= 11:
                try:
                    check_digit = int(corrected_text[10])
                    calc = iso6346_check_digit(corrected_text[:10])
                    checksum_valid = (calc == check_digit)
                except:
                    checksum_valid = False

            # Visualize
            # draw ROI
            cv2.rectangle(img, (x1p,y1p), (x2p,y2p), (0,255,0), 2)
            cv2.putText(img, f"{roi['label']} {corrected_text}", (x1p, max(12, y1p-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

            for c in corrected_chars:
                bx = list(map(int, c['bbox_abs']))
                cv2.rectangle(img, (bx[0],bx[1]), (bx[2],bx[3]), (255,0,0), 1)
                cv2.putText(img, c['corrected_label'], (bx[0], max(12,bx[1]-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

            # Save ROI raw/merged debug visuals if required
            if DEBUG:
                # draw merged boxes on roi_for_vis
                debug_roi = roi_for_vis.copy()
                for m in merged:
                    b = list(map(int, m['bbox']))
                    cv2.rectangle(debug_roi, (b[0],b[1]), (b[2],b[3]), (0,0,255), 1)
                    cv2.putText(debug_roi, f"{m['label']}:{m['confidence']:.2f}", (b[0], b[1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                cv2.imwrite(f"debug_roi_{i}.jpg", debug_roi)

            # Pack JSON
            output['rois'].append({
                'format': roi['label'],
                'roi_bbox': [x1p,y1p,x2p,y2p],
                'confidence': roi['confidence'],
                'raw_text': "".join([c['label'] for c in raw_chars]),
                'corrected_text': corrected_text,
                'check_digit': check_digit,
                'checksum_valid': checksum_valid,
                'characters': [{
                    'label': c['label'],
                    'corrected_label': c.get('corrected_label', c['label']),
                    'confidence': float(c['confidence']),
                    'bbox_abs': c['bbox_abs'],
                    'members': [{
                        'label': m['label'],
                        'confidence': float(m['confidence']),
                        'bbox': m['bbox']
                    } for m in c.get('members', [])]
                } for c in corrected_chars]
            })

        # Save outputs
        cv2.imwrite(out_image_path, img)
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        if DEBUG:
            print("Output saved to", out_image_path, out_json_path)

# ================================================
# Запуск
# ================================================
if __name__ == "__main__":
    recognizer = ISO6346AdvancedRecognizer(ROI_MODEL_PATH, CHAR_MODEL_PATH)
    recognizer.process_image(r"C:\Users\User\Desktop\Projects\Work\ISO_6346_rec\test_img\1-123808001-OCR-RF-D01_jpg.rf.00dc11c689f9c7178cc65cdecdb7b37d.jpg", "result_merged.jpg", "result_merged.json", pad=10)
