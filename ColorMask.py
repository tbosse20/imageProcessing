import cv2
import numpy as np

def colorMask(img: np.ndarray, lower: tuple, upper: tuple) -> np.ndarray:
    # Convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow('Mask', mask)

    # Slice the color
    imask = mask > 0
    maskedImg = np.zeros_like(img, np.uint8)
    maskedImg[imask] = img[imask]

    return maskedImg

if __name__ == '__main__':
    image = cv2.imread('TestImages/Grass.JPEG')
    colorMask(image, (31, 144, 109), (53, 255, 184))
    #colorMask(image, (40, 0, 104), (72, 255, 212))
    cv2.waitKey(0)