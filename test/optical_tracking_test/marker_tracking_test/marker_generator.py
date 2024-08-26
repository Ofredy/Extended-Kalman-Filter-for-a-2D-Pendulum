import cv2
import numpy as np

# Define parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250) # Choose a dictionary, e.g., 6x6 with 250 markers
marker_size = 200  # Size of the marker in pixels
border_bits = 1  # Size of the border around the marker
num_markers = 4  # Number of markers to generate

# Generate and save each marker
for marker_id in range(num_markers):
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_image, border_bits)
    
    # Save the marker to a file
    marker_filename = f"aruco_marker_{marker_id}.png"
    cv2.imwrite(marker_filename, marker_image)
    print(f"Marker {marker_id} saved as {marker_filename}")

    # Optionally, display the marker
    cv2.imshow(f'Marker {marker_id}', marker_image)
    cv2.waitKey(1)  # Wait for a key press to close the image window

cv2.destroyAllWindows()