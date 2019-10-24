# template matching
import cv2
import numpy as np


def get_hist_dist(im1, im2):
    h1 = cv2.calcHist([im1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([im2], [0], None, [256], [0, 256])
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


def show_img(img1):
    cv2.imshow("Image", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


template = cv2.imread('images/church.jpg')
cropped = cv2.imread('images/croppedChurch.jpg')
w, h = cropped.shape[0:2]
result = {}

print("Chargement...")

for i in range(template.shape[0] - w):
    for j in range(template.shape[1] - h):
        currentImagePart = template[i:i + w, j:j + h]
        res = get_hist_dist(currentImagePart, cropped)
        if res > 0.98:
            cv2.rectangle(template, (j, i), (j + h, i + w), (0, 255, 255), 2)

show_img(template)