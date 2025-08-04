
"""
import cv2

# 1. Create a VideoCapture object
# The argument '0' selects the default camera. If you have multiple cameras,
# you might need to try '1', '2', etc.
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# 2. Loop to continuously read frames from the camera
while True:
    # Read a new frame from the video capture
    # 'ret' is a boolean that is True if the frame was read successfully
    # 'frame' is the actual image frame captured
    ret, frame = cap.read()

    # If the frame was not captured correctly, we break the loop
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # 3. Display the resulting frame
    cv2.imshow('PC Camera Feed', frame)

    # 4. Wait for a key press and exit on 'q'
    # cv2.waitKey(1) waits for 1 millisecond. This is crucial for the video to play.
    # The '0xFF == ord('q')' part checks if the pressed key was 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
"""

import os
import numpy as np
from PIL import Image


def images_to_numpy(folder_path):
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(( ".jpg"))]
    image_files.sort()  # sort for consistent ordering
    
    images = []
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        # Using PIL
        img = Image.open(img_path)
            
        img_array = np.array(img)
        images.append(img_array)
    
    # Convert list of arrays to single numpy array
    images_array = np.stack(images, axis=0)
    
    return images_array

# Example usage:
folder_path = 'path/to/your/images'
all_images = images_to_numpy(folder_path, target_size=(128, 128))  # resize all to 128x128

print(f"Shape of the resulting array: {all_images.shape}")
print(f"Data type: {all_images.dtype}")
