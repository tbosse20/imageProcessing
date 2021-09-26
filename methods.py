import cv2
import numpy as np

def colorTreshold(img, lower, upper):
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)

    ## slice the green
    imask = mask > 0
    maskedImg = np.zeros_like(img, np.uint8)
    maskedImg[imask] = img[imask]

    return maskedImg

def tresholdPosition(maskedImg):
    pass