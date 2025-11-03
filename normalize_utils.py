import cv2
import numpy as np

def normalize_char(ch, size=(40, 60)):
    h, w = ch.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(ch, (new_w, new_h))

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_off = max((target_w - new_w) // 2, 0)
    y_off = max((target_h - new_h) // 2, 0)
    h_end = min(y_off + new_h, target_h)
    w_end = min(x_off + new_w, target_w)
    resized_cropped = resized[:h_end - y_off, :w_end - x_off]

    canvas[y_off:h_end, x_off:w_end] = resized_cropped
    return canvas
