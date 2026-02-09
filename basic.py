import cv2 as cv
img = cv.imread(r"D:\\Coding++\\web_dev_and_projects\\open_cv\\assets\\capybara.jpeg")

# converting to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# converting a blur image
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)

# edge cascade
edging = cv.Canny(blur, 125, 175)

# dilating the image
dilated = cv.dilate(edging, (7, 7), iterations=1)

# eroding
eroded = cv.erode(dilated, (7, 7), iterations=1)

# resize and crop
resized = cv.resize(img, (700, 700), interpolation=cv.INTER_CUBIC)
cropped = resized[50:500, 100:500]

# cv.imshow('capybara', img)
cv.imshow('capybara', resized)
cv.imshow('capybara cropped', cropped)
# cv.imshow('capybara gray', gray)
# cv.imshow('blurred capybara', blur)
cv.imshow("edgy capybara", edging)
cv.imshow("dilated capybara", dilated)
cv.imshow("eroded capybara", eroded)


cv.waitKey(10000)