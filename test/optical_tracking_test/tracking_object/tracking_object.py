import cv2
import numpy as np

# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Change 0 to the video file path if using a pre-recorded video

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take the first frame and convert it to grayscale
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detect the initial position of the pendulum (for example, using a point in the frame)
# Use manual selection of a point of interest for simplicity
x, y, w, h = cv2.selectROI("Select Pendulum", old_frame, fromCenter=False, showCrosshair=True)
pendulum_point = np.array([[x + w//2, y + h//2]], dtype=np.float32).reshape(-1, 1, 2)

cv2.destroyWindow("Select Pendulum")

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture video")
        break

    # Convert the frame to grayscale
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow using the Lucas-Kanade method
    new_point, status, error = cv2.calcOpticalFlowPyrLK(old_gray, grey_frame, pendulum_point, None, **lk_params)

    # If the point is found (status == 1)
    if status[0] == 1:
        # Draw the tracks
        a, b = new_point.ravel()
        c, d = pendulum_point.ravel()

        a, b, c, d = int(a), int(b), int(c), int(d)

        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

        # Update the previous point
        pendulum_point = new_point

    # Overlay the tracks on the frame
    img = cv2.add(frame, mask)

    # Display the result
    cv2.imshow('Pendulum Tracking', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the previous frame and previous points
    old_gray = grey_frame.copy()

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()