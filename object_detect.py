import cv2
import numpy as np

# Video Input
cap = cv2.VideoCapture(r"D:\\Coding++\\web_dev_and_projects\\open_cv\\assets\\sample_footage2.mp4")

ret, prev_frame = cap.read()
if not ret:
    print("Error reading video")
    exit()
print("Video is working")

prev_frame = cv2.resize(prev_frame, (640, 480))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)


# Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=50,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MIN_AREA = 400


# Feature Detection (for stabilization)
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=30,
    blockSize=3
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)


# Main Working Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # STABILIZATION STEP
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None, **lk_params
    )

    valid_prev = prev_pts[status == 1]
    valid_curr = curr_pts[status == 1]

    if len(valid_prev) > 10:
        m, _ = cv2.estimateAffinePartial2D(valid_curr, valid_prev)
        if m is not None:
            stabilized = cv2.warpAffine(frame, m, (640, 480))
        else:
            stabilized = frame.copy()
    else:
        stabilized = frame.copy()

    # Update for next frame
    prev_gray = gray.copy()
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

    # MOTION DETECTION (on stabilized frame)
    raw_mask = fgbg.apply(stabilized)


    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    display = stabilized.copy()

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(largest)

            # Draw bounding box
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Compute center coordinates
            cx = x + w // 2
            cy = y + h // 2

            # Draw center point
            cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

            # Display coordinates on frame
            coord_text = f"X: {cx}, Y: {cy}"
            cv2.putText(
                display,
                coord_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # OPTIONAL: print coordinates to terminal
            print(f"Object center at: ({cx}, {cy})")


    # DISPLAY
    cv2.imshow("Original (Shaky)", frame)
    cv2.imshow("Stabilized Frame", stabilized)
    cv2.imshow("Foreground Mask", clean_mask)
    cv2.imshow("Final Detection", display)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
