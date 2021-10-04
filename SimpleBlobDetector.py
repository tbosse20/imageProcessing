import cv2
import numpy as np
from ColorMask import colorMask

def SimpleBlobDetector(image, window_name, lower, upper):

    # Read image
    originalImage = image.copy()                                                        # Make a copy of the image
    image = colorMask(image, lower, upper)                                              # Threshold image with color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                     # Image to grayscale
    (thresh, image) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # Convert image to binary
    image = cv2.bitwise_not(image)                                                      # Invert image
    #image = cv2.Canny(image, 100, 200)                                                 # Filter only the edges of image

    cv2.imshow('Processed image', image)                                                # Show processed image

    # Blob detector
    params = cv2.SimpleBlobDetector_Params()

    # Blob detector - parameters
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

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(originalImage, keypoints, blank, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(number_of_blobs)
    cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 1)

    cv2.imshow('SimpleBlobDetector', blobs)

    return blobs