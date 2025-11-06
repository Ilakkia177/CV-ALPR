import cv2
import os
import numpy as np
import string

def load_templates(templates_path):

    # Used to categorize each template image later
    letters = list(string.ascii_uppercase)
    digits = list(string.digits)

    # to hold template images
    template_dict = {"letters": {}, "digits": {}}

    for file in os.listdir(templates_path):
        name_full = os.path.splitext(file)[0].upper() # output: A_1
        name = name_full.split("_")[0]  # output : A

        # read in grayscale
        img = cv2.imread(os.path.join(templates_path, file), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        # threshold using Otsu’s method to binarize the image automatically
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(img) > 127: # white background, invert it
            img = 255 - img

        # helps handle varying lighting conditions among templates    
        img = cv2.equalizeHist(img)

        # to ensure  same brightness scaling
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        if name in letters:
            template_dict["letters"].setdefault(name, []).append(img)
        elif name in digits:
            template_dict["digits"].setdefault(name, []).append(img)

    print(f" Loaded templates: {len(template_dict['letters'])} letters, {len(template_dict['digits'])} digits")
    return template_dict
