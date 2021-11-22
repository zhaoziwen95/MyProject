import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.image as mpimg
import cv2

def load_images_into_stack(folder = "", file_type = None, suffix = None):

    files = os.listdir(folder)
    image_list = []

    for file in files:
        if file_type is not None:
            if file_type not in file:
                continue
        if suffix is not None:
            if suffix not in file:
                continue

        image_list.append(mpimg.imread(os.path.join(folder, file)))

    return np.array(image_list)           # [idx_img, M, N, color_ch]; Attention: Indexing refers to matrices!



def mask_colored_region_hsv(img_stack, lower_hsv_bound = (30, 0.6, 0.3), upper_hsv_bound = (60, 1, 0.7)):

    masks_water = np.zeros(np.shape(img_stack)[:3])

    for i, img in enumerate(img_stack):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masks_water[i] = cv2.inRange(img_hsv, lower_hsv_bound, upper_hsv_bound)

    return masks_water