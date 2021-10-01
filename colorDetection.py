import cv2
import numpy as np
from BlobDetection import BlobDetection, Blob

def drawRectangle(blobs, frame, animate):
    detectedImage = frame.copy()
    for blob in blobs:
        position = (blob.x, blob.y)
        dimension = (blob.x + blob.w, blob.y + blob.h)
        color = (0, 0, 255)
        detectedImage = cv2.rectangle(detectedImage, position, dimension, color, 1)
        if animate:
            cv2.imshow('Show', detectedImage)
            cv2.waitKey(5)
    return detectedImage


def colorMask(img: np.ndarray, offset: float, lower: tuple, upper: tuple) -> np.ndarray:
    # Convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.subtract(lower, offset), np.add(upper, offset))

    # Slice the color
    imask = mask > 0
    maskedImg = np.zeros_like(img, np.uint8)
    maskedImg[imask] = img[imask]

    return maskedImg


def colorDetection(format: str, originalFrame: np.ndarray, blurLevel: int, offset: [int], lower: [int], upper: [int],
                   type: str,
                   lowerRes: int) -> [Blob]:
    # originalFrame = cv2.imread("", cv2.IMREAD_UNCHANGED)
    height, width, channels = originalFrame.shape

    imageProcess = originalFrame.copy()

    # TO-DO: LOWER/DECREASE RESOLUTION
    # print(width)
    imageProcess = cv2.resize(imageProcess, (int(width / lowerRes), int(height / lowerRes)))
    # cv2.imshow('imageProcess', imageProcess)
    # cv2.waitKey(0)

    # Blur image
    imageProcess = cv2.GaussianBlur(imageProcess, (blurLevel, blurLevel), 0)
    # cv2.imshow('Blur', imageProcess)

    # Mask water
    imageProcess = colorMask(imageProcess, offset, lower, upper)
    # cv2.imshow('Color mask', imageProcess)

    # Canny Edge Detection
    imageProcess = cv2.cvtColor(imageProcess, cv2.COLOR_BGR2GRAY)
    imageProcess = cv2.Canny(imageProcess, 100, 200)
    # cv2.imshow('Edge detection', imageProcess)

    # Blob detect image
    blobs = BlobDetection(imageProcess, '0').getAreas()

    if 1 < len(blobs): blobs = BlobDetection.mergeBlobs(blobs)

    # Remove to small blobs
    for blob in list(blobs):
        if blob.w < Blob.sizeFilter or blob.h < Blob.sizeFilter:
            blobs.remove(blob)

    for blob in blobs:
        blob.type = type

    for blob in blobs:
        blob.x = blob.x * lowerRes
        blob.y = blob.y * lowerRes
        blob.w = blob.w * lowerRes
        blob.h = blob.h * lowerRes

    imageProcess = drawRectangle(blobs, originalFrame, False)
    # cv2.imshow('imageProcess', imageProcess)
    # cv2.waitKey(1)

    if format == 'video':
        return imageProcess
    elif format == 'pic':
        cv2.imshow('Find ' + type, imageProcess)
        cv2.waitKey(0)


def videoTestNewMethod(image, window_name, lowerThresh, upperThresh):
    # Read image
    originalImage = image.copy()
    image = colorMask(image, (0, 0, 0), (48, 30, 104), (78, 122, 255))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, image) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    image = cv2.bitwise_not(image)
    #image = cv2.Canny(image, 100, 200) # Edge
    cv2.imshow('Processed image', image)
    # Set up the detector with default parameters.
    # Initialize parameter settiing using cv2.SimpleBlobDetector

    # Set our filtering parameters
    # Initialize parameter settiing using cv2.SimpleBlobDetector

    params = cv2.SimpleBlobDetector_Params()

    params.filterByColor = False
    params.minThreshold = 10
    params.maxThreshold = 30
    params.blobColor = 0
    params.minArea = 100
    params.maxArea = 500000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.minCircularity = 100
    params.maxCircularity = 1000

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(image)

    # print(len(keypoints))
    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(originalImage, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    return blobs


def videoTest():
    cap = cv2.VideoCapture('TestImages/greensmall.mp4')

    frameCount = 0

    while (1):
        # Capture frame-by-frame
        _, frame = cap.read()
        frame = cv2.resize(frame, (360, 640))

        frameCount += 1

        # if frameCount % 5 == 0:
        """
        blurLevel = 9
        offset = (10, 0, 0)
        lower = (48, 30, 104)
        upper = (78, 122, 255)
        type = 'greenGlove'
        lowerRes = 2
        frame = colorDetection('video', frame, blurLevel, offset, lower, upper, type, lowerRes)
        """
        frame = videoTestNewMethod(frame, 'Window', 0, 0)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()

    cv2.destroyAllWindows()


def imageTest(window_name, lowerThresh, upperThresh):
    image = cv2.imread("TestImages/Gloves1.png")
    image = videoTestNewMethod(image, window_name, lowerThresh, upperThresh)

    # Show image
    cv2.imshow(window_name, image)
    cv2.waitKey(1)

def nothing(value):
    pass

window_name = 'test'
cv2.namedWindow(window_name)
cv2.createTrackbar('Lower threshold', window_name, 1, 255, nothing)
cv2.createTrackbar('Upper threshold', window_name, 1, 255, nothing)

""" """
#videoTest()
#while True:

# Getting input from sliders
lowerThresh = cv2.getTrackbarPos('Lower threshold', window_name)
upperThresh = cv2.getTrackbarPos('Upper threshold', window_name)

# colorDetection('pic', cv2.imread('TestImages/KingDomino1.jpg'), 9, (0, 0, 0), (100, 154, 28), (158, 255, 215), 'Water', 1)
# colorDetection('pic', cv2.imread('TestImages/KingDomino1.jpg'), 9, (0, 0, 0), (40, 0, 104), (72, 255, 212), 'Grass', 1)
# colorDetection('pic', cv2.imread('TestImages/KingDomino1.jpg'), 9, (10, 0, 0), (43, 52, 12), (95, 238, 97), 'Forest', 1)
# colorDetection('pic', cv2.imread('TestImages/KingDomino1.jpg'), 9, (0, 0, 0), (24, 68, 95), (36, 139, 171), 'Dirt', 1)
# colorDetection('pic', cv2.imread('TestImages/Gloves1.png'), 9, (0, 0, 0), (48, 30, 104), (78, 122, 255), 'greenGlove', 1)

#imageTest(window_name, lowerThresh, upperThresh)