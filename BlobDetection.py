from GrassFireMethod import GrassFireMethod, Area
import cv2
from ColorMask import colorMask
import numpy as np

class Blob(Area):
    def __init__(self, grass, x: int, y: int):
        super().__init__()
        self.x = x
        self.y = y
        self.w = 0
        self.h = 0
        Blob.blobThreshold = 0
        Blob.sizeFilter = 10

    def setThreshold(blobThreshold):
        Blob.blobThreshold = blobThreshold


class BlobDetection(GrassFireMethod):
    def __init__(self, matrix, fireMethod: str):
        super().__init__(matrix, Blob, fireMethod)

    def condition(self, Blob, grass: int) -> bool: return 200 < grass

    def handle(self, blob: Blob, grass, x: int, y: int):
        blob.x = min(x, blob.x)
        blob.w = max(x - blob.x, blob.w)
        blob.y = min(y, blob.y)
        blob.h = max(y - blob.y, blob.h)
        self.matrix[y][x] = 50

        return blob, grass

    def getAreas(self) -> list:
        return self.areas

    def mergeBlobs(blobs):
        blobs = list(set(blobs))
        for blob1 in list(blobs):
            for blob2 in list(blobs):
                if blob1 is not blob2:
                    if checkOverLap(blob1, blob2):
                        blob2.x = min(blob1.x, blob2.x)
                        blob2.w = max(blob1.w, blob2.w)
                        blob2.y = min(blob1.y, blob2.y)
                        blob2.h = max(blob1.h, blob2.h)
                        if blob1 in blobs:
                            blobs.remove(blob1)
        return blobs


def checkOverLap(obj1, obj2) -> bool:
    if obj1.x - Blob.blobThreshold < obj2.x + obj2.w:
        if obj1.x + obj1.w + Blob.blobThreshold > obj2.x:
            if obj1.y - Blob.blobThreshold < obj2.y + obj2.h:
                if obj1.h + obj1.y + Blob.blobThreshold > obj2.y:
                    return True
    return False


def blobDetection(format: str, originalFrame: np.ndarray, blurLevel: int, lower: [int], upper: [int],
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
    imageProcess = colorMask(imageProcess, lower, upper)
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
    cv2.imshow('Self made blob detector', imageProcess)


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