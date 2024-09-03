import os
import cv2
import numpy as np

# Load camera calibration parameters
CAMERA_CALIBRATION_PATH = os.path.join('..', 'camera_calibration_test')
camera_matrix = np.load(os.path.join(CAMERA_CALIBRATION_PATH, 'camera_matrix.npy'))
dist_coeffs = np.load(os.path.join(CAMERA_CALIBRATION_PATH, 'dist_coeffs.npy'))

# Load the ArUco dictionary and set up the parameters for detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Choose a dictionary, e.g., 6x6 with 250 markers
parameters = cv2.aruco.DetectorParameters()

# Define known positions of markers (in meters)
marker_positions = {
    1: (0.0, 0.0),  # Marker 1 at origin
    3: (0.61, 0.61),  # Marker 3 at (0.61 meters, 0.61 meters)
}

# Start video capture from the webcam (0 is the default camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

########### Optical Flow Init ###########

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
        print("Error: Could not read frame.")
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Draw detected markers on the undistorted frame
    image_markers = cv2.aruco.drawDetectedMarkers(undistorted_frame.copy(), corners, ids)

    # Calculate Optical Flow using the Lucas-Kanade method
    new_point, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, pendulum_point, None, **lk_params)

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

    if ids is not None:
        # Initialize variables for marker centers
        marker_center_x_1, marker_center_y_1 = None, None
        marker_center_x_3, marker_center_y_3 = None, None

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_corners = marker_corners.reshape((4, 2))
            top_left = marker_corners[0]

            # Calculate the center of the marker
            marker_center_x = (top_left[0] + marker_corners[2][0]) / 2
            marker_center_y = (top_left[1] + marker_corners[2][1]) / 2

            if marker_id == 1:
                marker_center_x_1, marker_center_y_1 = marker_center_x, marker_center_y
            elif marker_id == 3:
                marker_center_x_3, marker_center_y_3 = marker_center_x, marker_center_y

            # Continue processing for other markers if needed

        if marker_center_x_1 is not None and marker_center_x_3 is not None:
            # Calculate pixel distance between Marker 1 and Marker 3
            pixel_distance = np.sqrt((marker_center_x_3 - marker_center_x_1) ** 2 +
                                     (marker_center_y_3 - marker_center_y_1) ** 2)

            # Calculate the conversion factor from pixels to meters
            real_world_distance = np.sqrt((marker_positions[3][0] - marker_positions[1][0]) ** 2 +
                                          (marker_positions[3][1] - marker_positions[1][1]) ** 2)
            conversion_factor = real_world_distance / pixel_distance

            # Calculate the relative position of the pendulum in pixels
            pendulum = pendulum_point[0][0]

            relative_position_x = pendulum[0] - marker_center_x_1
            relative_position_y = pendulum[1] - marker_center_y_1

            # Convert the relative position to meters
            relative_position_x_meters = relative_position_x * conversion_factor
            relative_position_y_meters = relative_position_y * conversion_factor

            print(f"Relative Position of Pendulum (meters): ({relative_position_x_meters}, {relative_position_y_meters})")

    # Display the result
    _ = cv2.add(frame, mask)
    _ = cv2.add(frame, image_markers)

    # Display the resulting frame
    cv2.imshow('Pendulum Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the previous frame and previous points
    old_gray = gray_frame.copy()

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
