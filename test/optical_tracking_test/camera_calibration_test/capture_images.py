import cv2
import os

# Settings
output_dir = 'calibration_images_1/'
chessboard_size = (9, 6)  # Adjust this to match your chessboard pattern
image_prefix = 'calib_image_'
image_format = '.jpg'
image_counter = 0

# Create the directory to store images if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the default camera (you can specify a different camera by changing the argument to `cv2.VideoCapture`)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Unable to open the camera.")
    exit()

print("Press 's' to save an image for calibration, and 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture an image.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # If found, draw and display the corners
    if ret:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)

    # Display the frame with detected corners (if found)
    cv2.imshow('Camera Feed', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Press 's' to save the image
        image_counter += 1
        image_filename = f"{output_dir}{image_prefix}{image_counter}{image_format}"
        cv2.imwrite(image_filename, frame)
        print(f"Saved {image_filename}")

    elif key == ord('q'):  # Press 'q' to quit the loop
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print("Image capture complete.")