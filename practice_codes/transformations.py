import cv2 as cv
import numpy as np

img = cv.imread(r"D:\\Coding++\\web_dev_and_projects\\open_cv\\assets\\capybara.jpeg")
cv.imshow('capybara', img)

resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC) 

def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]
    if (rotPoint is None):
        rotPoint = (width//2, height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimension = (width, height)
    return cv.warpAffine(img, rotMat, dimension)

def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimension = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)

rotated = rotate(img, 157)
rerotate = rotate(rotated, 35)
translated = translate(img, 100, 100)

cv.imshow('translated capybara', translated)
cv.imshow('rotated capybara', rotated)
cv.imshow('rerotated capybara', rerotate)
cv.imshow('resized capybara', resized)

cv.waitKey(10000)

