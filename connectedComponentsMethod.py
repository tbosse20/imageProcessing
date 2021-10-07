import cv2
import numpy as np
import imageProcessing.ProcessImage

def connectedComponentsMethod(originalImage, blur: int, lower, upper, type: str):

    processedImage = imageProcessing.ProcessImage.processImage(originalImage, lower, upper)

    num_labels, labels = cv2.connectedComponents(processedImage)

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    blobs = []
    for n in range(1, num_labels):
        nonZeroX, nonZeroY = np.where(labels[:,:] == n)

        x, y = min(nonZeroY), min(nonZeroX)
        w, h = max(nonZeroY) - x, max(nonZeroX) - y

        blobs.append(imageProcessing.ProcessImage.Blob(x, y, w, h))

    notFiltered = originalImage.copy()
    for blob in blobs:
        notFiltered = cv2.rectangle(notFiltered, (blob.x, blob.y), (blob.x + blob.w, blob.y + blob.h), (0, 0, 255), 1)

    cv2.imshow('Not filtered', notFiltered)

    blobs = imageProcessing.ProcessImage.filterBlobs(blobs, minWidth=0, minHeight=0)
    #blobs = imageProcessing.ProcessImage.mergeBlobs(blobs, 10)
    blobs = imageProcessing.ProcessImage.filterBlobs(blobs, minWidth=60, minHeight=60)

    for blob in blobs:
        originalImage = cv2.rectangle(originalImage, (blob.x, blob.y), (blob.x + blob.w, blob.y + blob.h), (0, 0, 255), 1)

    cv2.imshow('ConnectedComponentsMethod', originalImage)

    for blob in blobs:
        blob.type = type

    return blobs