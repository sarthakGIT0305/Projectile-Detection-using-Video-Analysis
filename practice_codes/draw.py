import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale = 0.50): # function to rescale frames or images
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# make blank box
blank = np.zeros((500, 500, 3), dtype = 'uint8')

# paint color
blank[100:200, 344:449] = 16, 87, 67

# draw rectangle
cv.rectangle(blank, (blank.shape[1]//5, 100), (blank.shape[1]//2, blank.shape[0]//2), (232, 43, 100), thickness = 3)

# draw a circle
cv.circle(blank, (blank.shape[1]//2 + 100, blank.shape[0]//2 - 50), 50, (40, 21, 113), thickness = cv.FILLED)

# draw a line
cv.line(blank, (10, 101), (100, 301), (122, 122, 32), thickness = 3)

# write text on image
cv.putText(blank, "helloworld", (225, 225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 222, 222), 1)

# img = cv.imread('assets/cat.jpg')
# img = rescaleFrame(img, 0.25)
# cv.imshow('billi', img)

cv.imshow('Blank', blank)

cv.waitKey(5000)