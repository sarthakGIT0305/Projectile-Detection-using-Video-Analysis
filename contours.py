import cv2 as cv
import numpy as np

img = cv.imread(r"D:\\Coding++\\web_dev_and_projects\\open_cv\\assets\\cat.jpg")
img = cv.resize(img, (img.shape[1]//9, img.shape[0]//9), interpolation=cv.INTER_CUBIC)
cv.imshow('cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

canny = cv.Canny(img, 125, 175)
cannygray = cv.Canny(gray, 125, 175)

cv.imshow('canny cat', canny)
cv.imshow('canny gray cat', cannygray)
cv.imshow('gray cat', gray)

cv.waitKey(7000)