import os
from crop import crop
import cv2
def getImageNameList(dataAddress):
    """get the filenames in your dataset
        return a list"""
    list = []
    for f in os.listdir(dataAddress):
        if f.endswith('.png'):
            list.append(f)
    return list

def getImage(i,dataAddress,list):
    """get a picture and crop it,
        return this picture"""
    img = cv2.imread(dataAddress + '/' + list[i])
    img = crop(img)
    # cv2.imshow('original', img)
    # cv2.waitKey(1)
    return img
