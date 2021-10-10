import cv2
import numpy as np
import matplotlib.pyplot as plt


# MISSING: CAN FIND THRESHOLD BUT NOT WITH MULTIPLE BUMPS
def getColorThreshold(img):

    # Show image for good measures
    cv2.imshow('Filter image', img)

    # Make mask to ignore fully white #
    #mask = img.copy()
    # Use all pixels not including 255 values
    mask = np.all((img[:, :] < 255) & (img[:, :] > 0), 2)
    # Convert bool to int
    mask = mask.astype(np.uint8, copy=False)
    # Change 1's to 255
    mask[mask == 1] = 255
    # Show mask for good measures
    cv2.imshow('Mask', mask)

    # Get the histogram for each HSV channel #
    hist = []
    # Convert image to HSV channels
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Separate HSV channels
    HSV = [img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]]
    for v in HSV:
        # Get histogram from each channel
        hist.append(cv2.calcHist([v], [0], mask, [256], [0, 256]))

    """ Show histogram    
    rgb = ['r', 'g', 'b']
    hsv = ['h', 's', 'v']
    for v, label in enumerate(hist):
        plt.plot(hist[v], color=rgb[v], label=hsv[v])
    plt.legend()
    plt.show()
        """

    # Find upper and lower #
    lower, upper = [], []
    for x, values in enumerate(hist):
        # Flatten all values into one list
        flat_list = np.array([int(i) for i in values])
        # Get values above mark threshold (10 set, MISSING: needs to be coded)
        thresholdMarksValues = np.where(flat_list > 10)
        # Get minimum value of mark threshold
        minValue = min(thresholdMarksValues[0])
        lower.append(minValue)
        # Get maximum value of mark threshold
        maxValue = max(thresholdMarksValues[0])
        upper.append(maxValue)
    lower = np.array(lower)
    upper = np.array(upper)
    print(lower, upper)



    return [lower, upper]

def colorMask(originalImage: np.ndarray, type: str) -> np.ndarray:
    path = str('King Domino dataset/Colordata/Image1/' + type + '.JPEG')
    if __name__ == '__main__':
        path = '../' + path

    maskImage = cv2.imread(path)
    cv2.imshow('Original image', originalImage)

    thresholdMark = getColorThreshold(maskImage)

    return colorMaskManuel(originalImage, thresholdMark[0], thresholdMark[1])

def colorMaskManuel(originalImage: np.ndarray, lower: tuple, upper: tuple) -> np.ndarray:

    # Convert to hsv
    hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('Mask', mask)

    # Slice the color
    imask = mask > 0
    maskedImg = np.zeros_like(originalImage, np.uint8)
    maskedImg[imask] = originalImage[imask]

    cv2.imshow('Final filtered image', maskedImg)

    return maskedImg

if __name__ == '__main__':
    image = cv2.imread('TestImages/1.jpg')
    #colorMask(image, (31, 144, 109), (53, 255, 184))
    #colorMask(image, (40, 0, 104), (72, 255, 212))
    #colorMask(image, 'Grass')
    colorMask(image, 'Forest')

    #img = cv2.imread('TestImages/Grass.JPEG')
    #getColorThreshold(img)
    cv2.waitKey(0)