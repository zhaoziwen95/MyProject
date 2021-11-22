from crop import crop
from thresholdBGR import thresholdBGR
import numpy as np
import cv2


def mophology(img):
    """erosion and Dilation
        return a picture"""
    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((12, 12), np.uint8)
    img = cv2.erode(img, kernel1)
    img = cv2.dilate(img, kernel2)
    # cv2.imshow('morphology' + str(i), imgThreshold)
    # cv2.waitKey()
    return img


def processImg(img):
    """import a img ,crop it, find the water and than return image whit 800*800*3 """
    img_old = crop(img)
    img_new = thresholdBGR(img_old)
    img_new = mophology(img_new)
    temp = np.zeros([800, 800, 3], np.uint8)
    temp[:, :, 0] = img_new
    temp[:, :, 1] = img_new
    temp[:, :, 2] = img_new
    img_new = temp
    img=np.concatenate((img_old,img_new),axis=1)
    return img
