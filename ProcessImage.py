import cv2
import numpy as np

class Blob:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def processImage(image):

    # Process image
    # Make a copy of the image
    originalImage = image.copy()
    # Threshold image with color
    # Image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert image to binary
    """
    (thresh, image) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert image
    image = cv2.bitwise_not(image)
    # Filter only the edges of image
    # image = cv2.Canny(image, 100, 200)
    # Morph image
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    th, im_th = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV);
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    image = im_th | im_floodfill_inv
    """

    # Show processed image
    cv2.imshow('Processed image', image)

    return image

def checkOverLap(obj1, obj2, threshold: int) -> bool:
    if obj1.x - threshold < obj2.x + obj2.w:
        if obj1.x + obj1.w + threshold > obj2.x:
            if obj1.y - threshold < obj2.y + obj2.h:
                if obj1.h + obj1.y + threshold > obj2.y:
                    return True
    return False

def mergeBlobs(blobs: Blob, threshold: int) -> [Blob]:
    blobs = list(set(blobs))
    for blob1 in list(blobs):
        for blob2 in list(blobs):
            if blob1 is not blob2:
                if checkOverLap(blob1, blob2, threshold):
                    blob2.x = min(blob1.x, blob2.x)
                    blob2.w = max(blob1.w, blob2.w)
                    blob2.y = min(blob1.y, blob2.y)
                    blob2.h = max(blob1.h, blob2.h)
                    if blob1 in blobs:
                        blobs.remove(blob1)
    return blobs

def filterBlobs(blobs: Blob, minWidth, minHeight) -> [Blob]:
    filteredBlobs = []
    for blob in blobs:
        if blob.w > minWidth and blob.h > minHeight:
            filteredBlobs.append(blob)
            continue

    return filteredBlobs