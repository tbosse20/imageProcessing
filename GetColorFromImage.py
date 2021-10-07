import cv2
import numpy as np
import matplotlib.pyplot as plt

def getColorFromImage(img):
    cv2.imshow('Image', img)

    mask = img.copy()
    mask = np.all(img[:, :] != 255, 2)
    mask = mask.astype(np.uint8, copy=False)
    mask[mask == 1] = 255
    cv2.imshow('Mask', mask)

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
    hist_h = cv2.calcHist([h], [0], mask, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], mask, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], mask, [256], [0, 256])

    # Find upper and lower


    minH = min(i for i in hist_h if i > 10)
    print(minH)

    """ Show histogram """
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    plt.legend()
    plt.show()

    #return (lower, upper)

if __name__ == '__main__':
    img = cv2.imread('TestImages/Grass.JPEG')
    getColorFromImage(img)
    cv2.waitKey(0)