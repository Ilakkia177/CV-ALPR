import cv2
import numpy as np
from normalize_utils import normalize_char

def extract_plate_text(plate_roi, template_dict):

    # grayscale simplifies later processing while preserving character contrast
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)


    # Enhance local contrast using CLAHE(Contrast Limited Adaptive Histogram Equalization)helps recover faint characters under uneven lighting, shadows etc
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)


    # Use a horizontal rect kernel for top-hat filtering to highlight bright text strokes against darker background.
    # Adding the tophat result enhances small bright regions (edges of characters).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    gray = cv2.add(gray, tophat)


    # mild Gaussian blur to suppress noise before sharpening
    # Then use unsharp masking (addWeighted) to boost edge contrast
    # making character boundaries more pronounced
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    alpha = 1.3
    sharpened = cv2.addWeighted(gray, 1 + alpha, blurred, -alpha, 0)

    # normalize to make intensity range to have consistent brightness
    sharpened = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 17, 5) #converts to binary, inv
    


    #  fills small gaps in strokes and merges broken edge.improving continuity of character shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)


    # Compute vertical projection profile: sum of white pixels per column, helps find character boundaries (low-sum valleys = gaps between characters).
    projection = np.sum(morph, axis=0)
    mean_proj = np.mean(projection)
    valleys = np.where(projection < mean_proj * 0.6)[0] # possible char splits, tested, can be tuned

    # Group consecutive valley indices to identify meaningful split points
    splits = []
    if len(valleys) > 0:
        group = [valleys[0]]
        for i in range(1, len(valleys)):
            if valleys[i] - valleys[i - 1] > 2:
                splits.append(int(np.mean(group)))
                group = [valleys[i]]
            else:
                group.append(valleys[i])
        splits.append(int(np.mean(group)))

    # Skip leading flat projection area (non-text region on plate)
    flat_threshold = 0.08 * np.max(projection)
    min_valid_start = np.argmax(projection > flat_threshold)
    skip_start = max(min_valid_start - 3, 0)

    # Segment characters between valley gaps
    chars = []
    prev = skip_start
    for s in splits:
        if s <= skip_start:
            continue
        seg = morph[:, prev:s]
        if seg.shape[1] > 5:
            chars.append(seg)
        prev = s
    seg = morph[:, prev:]
    if seg.shape[1] > 5: # ignore too-narrow segments
        chars.append(seg)

    # Remove abnormally narrow first char -artifact
    if len(chars) > 1:
        widths = [ch.shape[1] for ch in chars]
        if widths[0] < 0.3 * np.mean(widths) or widths[0] < 10:
            chars = chars[1:]

    #lp-hong kong 4-6 charcaters only
    num_chars = len(chars)
    if num_chars < 4 or num_chars > 6:
        return None

    plate_text = ""
    for idx, ch in enumerate(chars):
        # ignore tiny or malformed character blobs.
        if ch.shape[0] < 10 or ch.shape[1] < 5:
            continue

        # normalize character to fixed size for template matching consistency.
        ch = normalize_char(ch, (50, 60))

        # binarize again using Otsu — ensures clean binary input for template match.
        _, ch = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(ch) > 127:
            ch = 255 - ch


 
        # compare each segmented char with stored templates
        # aplabet first digits later-use this logic to avoid FP
        best_score, best_match = -1, None
        group_type = "letters" if idx < 2 else "digits"
        for name, tmpl_list in template_dict[group_type].items():
            for tmpl in tmpl_list:
                tmpl = normalize_char(tmpl, (50, 60))
                res = cv2.matchTemplate(ch, tmpl, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(res)
                if score > best_score:
                    best_score, best_match = score, name
        if best_score > 0.2 and best_match:
            if best_match == 'I':
                best_match = 'T'      # hong kong plates have no I,O,Q
            elif best_match in ['O', 'Q']:
                best_match = '0'
            plate_text += best_match
    return plate_text if plate_text else None
