import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use the appropriate camera index (0 for the default webcam)

# Load the ArUco dictionary and set up the parameters for detection
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) # Choose a dictionary, e.g., 6x6 with 250 markers
parameters = cv2.aruco.DetectorParameters()

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
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
            print(f"Marker ID: {marker_id}, Position: ({center_x}, {center_y})")

            # Draw the ID and position on the frame
            cv2.putText(frame, f"ID: {marker_id}", (center_x - 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow('ArUco Marker Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()