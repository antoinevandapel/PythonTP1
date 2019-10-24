# template matching
import cv2
import numpy as np


def get_hist_dist(im1, im2):
    h1 = cv2.calcHist([im1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([im2], [0], None, [256], [0, 256])
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


ensemble_img = ['beach.jpg', 'dog.jpg', 'polar.jpg', 'bear.jpg', 'lake.jpg', 'moose.jpg']
template = cv2.imread('images/waves.jpg')
results = {}

for file in ensemble_img:
    img_rgb = cv2.imread('images/' + file)
    res = get_hist_dist(img_rgb, template)
    results[file] = res

imgResult = cv2.imread('images/' + min(results))
cv2.imshow("resultat", np.concatenate((template, imgResult), axis=1))
print(min(results))
cv2.waitKey(0)
