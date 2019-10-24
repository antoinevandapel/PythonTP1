import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/bgr.png')
cv2.imshow('All colors', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

blues = img[:, :, 0]
greens = img[:, :, 1]
reds = img[:, :, 2]

cv2.imshow('Blues', blues)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Greens', greens)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Reds', reds)
cv2.waitKey(0)
cv2.destroyAllWindows()

no_blues = img.copy()
no_blues[:, :, 0] = 0
cv2.imshow('No blues', no_blues)
cv2.waitKey(0)
cv2.destroyAllWindows()

no_greens = img.copy()
no_greens[:, :, 1] = 0
cv2.imshow('No greens', no_greens)
cv2.waitKey(0)
cv2.destroyAllWindows()

no_reds = img.copy()
no_reds[:, :, 2] = 0
cv2.imshow('No Reds', no_reds)
cv2.waitKey(0)
cv2.destroyAllWindows()