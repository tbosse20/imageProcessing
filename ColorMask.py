import cv2
import numpy as np
import matplotlib.pyplot as plt



def getColorThreshold(img):

    # Show image for good measures
    cv2.imshow('Image', img)

    # Make mask to ignore fully white #
    mask = img.copy()
    # Use all pixels not including 255 values
    mask = np.all(img[:, :] != 255, 2)
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

    # Find upper and lower #
    lower, upper = [], []
    for x, values in enumerate(hist):
        # Flatten all values into one list
        flat_list = np.array([int(i) for i in values])
        # Get values above mark threshold
        thresholdMarksValues = np.where(flat_list > 300)
        # Get minimum value of mark threshold
        minValue = min(thresholdMarksValues[0])
        # Get maximum value of mark threshold
        maxValue = max(thresholdMarksValues[0])
        lower.append(minValue)
        upper.append(maxValue)

        """
        # Get the index of the two lowest values # MISSING: NOT FINDING ANYTHING
        minIndex = np.where(flat_list == minValue)[0]
        maxIndex = np.where(flat_list == maxValue)[0]
        print(minIndex, maxIndex)

        #print(flat_list)
        gradient = np.gradient(flat_list)
        gradient = [int(i) for i in gradient]
        #print(gradient)
        # Get all values with minimum value above given threshold
        #cropped = [i for i in flat_list if i > 5]
        #print(cropped)
        #minIndex = np.where((abs(gradient) > 20) & (abs(gradient) < 50))
        #minIndex = gradient[(np.where((gradient >= 20) & (gradient <= 30)))]
        #print(minIndex)

        # Get steppness
        """

        """
        # Get the index of the minimum value
        minIndex = min(thresholdMarksIndex)
        # Append index to lower threshold
        lower.append(minIndex)

        # Get the index of the maximum value
        maxIndex = max(thresholdMarksIndex)
        # Append index to upper threshold
        upper.append(maxIndex)
        """

    lower = np.array(lower)
    upper = np.array(upper)
    print(lower, upper)

    """ Show histogram 
    rgb = ['r', 'g', 'b']
    hsv = ['h', 's', 'v']
    for v, label in enumerate(hist):
        plt.plot(hist[v], color=rgb[v], label=hsv[v])
    plt.legend()
    plt.show()
    """

    return [lower, upper]

def colorMask(originalImage: np.ndarray, type: str) -> np.ndarray:
    path = str('King Domino dataset/Colordata/Image1/' + type + '.JPEG')
    if __name__ == '__main__':
        path = '../' + path

    maskImage = cv2.imread(path)
    thresholdMark = getColorThreshold(maskImage)
    #return colorMask(img, threshold[0], threshold[1])
    cv2.imshow('Original image', originalImage)

#def colorMask(img: np.ndarray, lower: tuple, upper: tuple) -> np.ndarray:

    # Convert to hsv
    hsv = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, thresholdMark[0], thresholdMark[1])

    cv2.imshow('Mask ' + type, mask)

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