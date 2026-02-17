import cv2
import numpy as np
from collections import deque

# ==============================
# CONFIGURATION FLAGS
# ==============================

USE_ROI = True
USE_FRAME_DIFF = True
USE_MOG2 = True
USE_MASK_COMBINATION = True
USE_SPEED_FILTER = True
USE_MULTI_FRAME_VALIDATION = True

# ==============================
# VIDEO INPUT
# ==============================

cap = cv2.VideoCapture(r"D:\\Coding++\\web_dev_and_projects\\open_cv\\assets\\distance_3.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

ret, prev_frame = cap.read()
if not ret:
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) # conversion to greyscale

height, width = prev_frame.shape[:2]

# ==============================
# ROI SETTINGS (only upper region)
# ==============================

Percent_of_height = 1 # starting from top of the video
wall_line = int(height * Percent_of_height) 

# ==============================
# BACKGROUND SUBTRACTOR
# ==============================

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=30, # lower value -> more sensitive
    detectShadows=False
)

# ==============================
# MORPHOLOGY KERNEL
# ==============================

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# ==============================
# FILTER PARAMETERS
# ==============================

MIN_AREA = 1
MAX_AREA = 80000
MOTION_THRESHOLD = 5
VALIDATION_FRAMES = 5

trajectory = deque(maxlen=20)
validation_counter = 0
prev_center = None

# ==============================
# MAIN LOOP
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()

    # --------------------------------
    # STEP 1: ROI Restriction
    # --------------------------------
    if USE_ROI:
        roi = frame[0:wall_line, :]
    else:
        roi = frame

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # --------------------------------
    # STEP 2: Frame Differencing
    # --------------------------------
    if USE_FRAME_DIFF:
        prev_gray_roi = prev_gray[0:wall_line, :] if USE_ROI else prev_gray
        frame_diff = cv2.absdiff(prev_gray_roi, gray)
        _, diff_mask = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
    else:
        diff_mask = None

    # --------------------------------
    # STEP 3: Background Subtraction
    # --------------------------------
    if USE_MOG2:
        mog_mask = fgbg.apply(roi)
    else:
        mog_mask = None

    # --------------------------------
    # STEP 4: Combine Masks
    # --------------------------------
    if USE_MASK_COMBINATION and USE_FRAME_DIFF and USE_MOG2:
        combined_mask = cv2.bitwise_and(diff_mask, mog_mask)
    elif USE_FRAME_DIFF:
        combined_mask = diff_mask
    elif USE_MOG2:
        combined_mask = mog_mask
    else:
        combined_mask = np.zeros_like(gray)

    # --------------------------------
    # STEP 5: Morphological Cleaning
    # --------------------------------
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=2)

    # --------------------------------
    # STEP 6: Contour Detection
    # --------------------------------
    contours, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detected_object = False

    # if contours:
    #     largest = max(contours, key=cv2.contourArea)
    #     area = cv2.contourArea(largest)
    #     # --------------------------------
    #     # STEP 7: Size Filtering
    #     # --------------------------------
    #     if MIN_AREA < area < MAX_AREA:
    #         x, y, w, h = cv2.boundingRect(largest)
    #         # Adjust y if ROI used
    #         if USE_ROI:
    #             y_global = y
    #         else:
    #             y_global = y
    #         cx = x + w // 2
    #         cy = y_global + h // 2
    #         # --------------------------------
    #         # STEP 8: Speed Filtering
    #         # --------------------------------
    #         if USE_SPEED_FILTER and prev_center is not None:
    #             dx = abs(cx - prev_center[0])
    #             dy = abs(cy - prev_center[1])
    #             if dx + dy < MOTION_THRESHOLD:
    #                 detected_object = False
    #             else:
    #                 detected_object = True
    #         else:
    #             detected_object = True
    #         if detected_object:
    #             trajectory.append((cx, cy))
    #             prev_center = (cx, cy)
    #             validation_counter += 1
    #         else:
    #             validation_counter = 0
    #         # --------------------------------
    #         # STEP 9: Multi-frame Validation
    #         # --------------------------------
    #         if not USE_MULTI_FRAME_VALIDATION or validation_counter >= VALIDATION_FRAMES:
    #             cv2.rectangle(original_frame,
    #                           (x, y_global),
    #                           (x + w, y_global + h),
    #                           (0, 255, 0), 2)
    #             cv2.circle(original_frame, (cx, cy), 4, (0, 0, 255), -1)
    #             cv2.putText(original_frame,
    #                         f"X:{cx} Y:{cy}",
    #                         (x, y_global - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX,
    #                         0.5,
    #                         (0, 255, 0),
    #                         2)
    
    current_centers = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        # --------------------------------
        # STEP 7: Size Filtering
        # --------------------------------
        if MIN_AREA < area < MAX_AREA:

            x, y, w, h = cv2.boundingRect(cnt)

            y_global = y  # ROI already applied to top region

            cx = x + w // 2
            cy = y_global + h // 2

            valid_motion = True

            # --------------------------------
            # STEP 8: Speed Filtering
            # --------------------------------
            if USE_SPEED_FILTER and prev_center is not None:
                dx = abs(cx - prev_center[0])
                dy = abs(cy - prev_center[1])

                if dx + dy < MOTION_THRESHOLD:
                    valid_motion = False

            if valid_motion:
                current_centers.append((cx, cy))

                # --------------------------------
                # STEP 9: Multi-frame Validation
                # --------------------------------
                if not USE_MULTI_FRAME_VALIDATION or validation_counter >= VALIDATION_FRAMES:

                    cv2.rectangle(original_frame,
                                (x, y_global),
                                (x + w, y_global + h),
                                (0, 255, 0), 2)

                    cv2.circle(original_frame, (cx, cy),
                            4, (0, 0, 255), -1)

                    cv2.putText(original_frame,
                                f"X:{cx} Y:{cy}",
                                (x, y_global - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2)

    # Update previous center if at least one valid object detected
    if current_centers:
        prev_center = current_centers[0]  # or choose based on logic
        validation_counter += 1
    else:
        validation_counter = 0


    # Update previous frame
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --------------------------------
    # DISPLAY (Resize only for viewing)
    # --------------------------------
    resizing_factor = 0.5
    display_frame = cv2.resize(original_frame, (int(original_frame.shape[1] * resizing_factor), int(original_frame.shape[0] * resizing_factor)))
    display_mask = cv2.resize(clean_mask, ((int(clean_mask.shape[1] * resizing_factor), int(clean_mask.shape[0] * resizing_factor))))

    cv2.imshow("Detection", display_frame)
    cv2.imshow("Mask", display_mask)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
