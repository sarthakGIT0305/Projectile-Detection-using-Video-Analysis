import cv2 as cv

def rescaleFrame(frame, scale = 0.50): # function to rescale frames or images
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height): # work for live videos only
    capture.set(3, width)
    capture.set(4, height)
    

# images testing
# img = cv.imread("capybara.jpeg")
# cv.imshow('capybara', img)
# img2 = cv.imread('large_image.png')
# cv.imshow('trees', img2)

# video testing
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    # frame = rescaleFrame(frame, 2)
    newframe = rescaleFrame(frame, 2)
    cv.imshow('jane_doe', newframe)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()

# cv.waitKey(0)





