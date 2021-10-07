import cv2
import numpy as np



def SimpleBlobDetector(originalImage, processedImage):

    # Blob detector
    params = cv2.SimpleBlobDetector_Params()

    # Blob detector - parameters
    params.filterByColor = False
    params.minThreshold = 0
    params.maxThreshold = 255
    params.minArea = 10
    params.maxArea = 500000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.minCircularity = 100
    params.maxCircularity = 1000

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs
    keypoints = detector.detect(processedImage)
    # Draw blobs on our image as red circles
    blobs = cv2.drawKeypoints(originalImage, keypoints, 0, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Number of Circular Blobs:", str(len(keypoints)), str(str(len(keypoints) == 6)))

    cv2.imshow('SimpleBlobDetector', blobs)

    return keypoints
