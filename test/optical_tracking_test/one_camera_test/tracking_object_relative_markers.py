import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use the appropriate camera index (0 for the default webcam)


########### ArUco Init ###########

ARUCO_DX = 3.5 # ft
ARUCO_DY = 1.5 # ft

# Load the ArUco dictionary and set up the parameters for detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) # Choose a dictionary, e.g., 6x6 with 250 markers
parameters = cv2.aruco.DetectorParameters()

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

    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Convert the frame to grayscale (ArUco detection works better on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejected = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

    # If markers are detected
    if ids is not None:
        # Draw the markers on the frame
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Loop through each detected marker and print its ID and position
        for i in range(len(ids)):
            marker_id = ids[i][0]
            # Calculate the center of the marker
            corner_points = corners[i][0]
            center_x = int(corner_points[:, 0].mean())
            center_y = int(corner_points[:, 1].mean())
            
            # Print the marker ID and its center position
            #print(f"Marker ID: {marker_id}, Position: ({center_x}, {center_y})")

            # Draw the ID and position on the frame
            cv2.putText(frame, f"ID: {marker_id}", (center_x - 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Calculate Optical Flow using the Lucas-Kanade method
    new_point, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, pendulum_point, None, **lk_params)

    # temp code -> will only factor in two markers for now, eventually will do all 4 
     
    assert len(ids) == 4

    # marker 3 & 4 center_x
    m4_corner = corners[3][0]
    m4_center = int(m4_corner[:, 0].mean())

    m3_corner = corners[2][0]
    m3_center = int(m3_corner[:, 0].mean())

    camera_scale_x = ARUCO_DX / ( m4_center - m3_center )

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

        print("x coordinate: %.3f" %  (camera_scale_x*(a-x)))

    # Overlay the tracks on the frame
    img = cv2.add(frame, mask)

    # Display the resulting frame
    cv2.imshow('Pendulum Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update the previous frame and previous points
    old_gray = gray_frame.copy()

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
