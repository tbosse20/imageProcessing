from GrassFireMethod import GrassFireMethod, Area


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