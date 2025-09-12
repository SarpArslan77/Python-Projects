
import cv2
import json
import os
from cnn import ConvNet
import torch
import numpy as np
import torch.nn.functional as F
from collections import deque

MAIN_FILE_READ_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/"
json_file_path: str = os.path.join(MAIN_FILE_READ_PATH, "config.json")
best_model_parameters_path: str = os.path.join(MAIN_FILE_READ_PATH, "Best Model/best_model_parameters.pth")

# Load the configuration file.
with open(json_file_path, "r") as f:
    config = json.load(f)

# Unpack the parameters into variables for easy access.
camera_cfg = config["camera_params"]

camera_width: int = camera_cfg["cnn_input_width"]
camera_height: int = camera_cfg["cnn_input_height"]

# A list of the 7 emotions that the model can predict.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# A dictionary to map emotions to colors.
EMOTION_COLORS = {
    "Angry":    (0, 0, 255),     # Red
    "Disgust":  (0, 128, 0),     # Green
    "Fear":     (128, 0, 128),   # Purple
    "Happy":    (0, 255, 255),   # Yellow
    "Neutral":  (128, 128, 128),  # Gray
    "Sad":      (255, 0, 0),     # Blue
    "Surprise": (255, 165, 0),   # Orange
}

EMOTION_CLASSES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",

}

# Labels for the predictions.
def create_GUI(frame, predictions):

    # Convert tensor to a NumPy array.
    predictions = predictions.squeeze().numpy()

    # Create a black canvas for the dashboard.
    dashboard_width: int = 320
    dashboard = np.zeros((frame.shape[0], dashboard_width, 3), dtype="uint8")

    # Draw elements on the dashboard.
    cv2.putText(dashboard, "Emotion Predictions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
        bar_y: int = i * 40 + 80
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        
        # Format the text and draw it.
        text: str = f"{emotion}: {prob:.1%}"
        cv2.putText(dashboard, text, (10, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw the bar.
        bar_width: int = int(prob * (dashboard_width - 130))
        cv2.rectangle(dashboard, (180, bar_y + 5), (180 + bar_width, bar_y + 25), color, -1)

    # Stitch the camera frame and the dashboard together horizontally.
    combined_frame = np.hstack([frame, dashboard])

    return combined_frame

# Camera implementation

# Create the VideoCapture object.
camera = cv2.VideoCapture(0) # 0 = default camera

# Load the face detector outside the loop.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the cnn model and upload the parameters.
best_model = ConvNet()
checkpoint = torch.load(best_model_parameters_path)
best_model.load_state_dict(checkpoint["model_state_dict"])
best_model.eval()

# Keep a history of the last N predictions.
PREDICTION_HISTORY_LEN = 40
prediction_history = deque(maxlen=PREDICTION_HISTORY_LEN)

# Main loop to get continiues data.
print("\nCamera Opened! Press 'q' to quit.")
while True:
    ret, frame = camera.read()
    # ret: is a boolean value True if the frame was red successfully
    # frame: actual image captured

    # Flip the frame horizontally for a mirror-like view
    mirrored_frame = cv2.flip(frame, 1)
    # Turn into gray scale.
    gray_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
    # Detect faces.
    faces = faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
    # If there is no face detected, just keep the last result.
    if prediction_history:
        current_probabilities = prediction_history[-1]
    else:
        # If for whatever reason the list ist empty, just initialize it to zero.
        current_probabilities = torch.zeros((1, len(EMOTIONS)))

    # If a face is found, process it.
    if len(faces) > 0:
        # Get the first face found.
        (x, y, w, h) = faces[0]

        # Draw a rectangle around the detected face on the original frame.
        cv2.rectangle(mirrored_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop the face from the grayscaled frame.
        cropped_face = gray_frame[y:y+h, x:x+w]
        # Resize the cropped face to 48x48.
        resized_face = cv2.resize(cropped_face, (camera_width, camera_height))
        # Conver to tensor and normalize.
        tensor_frame = torch.from_numpy(resized_face).float()
        tensor_frame = (tensor_frame / 255.0 - 0.5) / 0.5
        # Add batch and channel dimensions. (48, 48) -> (1, 1, 48, 48)
        tensor_frame = tensor_frame.unsqueeze(0).unsqueeze(0)

        # Make predictions as emotion recognition.
        with torch.no_grad():
            # Get the raw logit scores from the model.
            output = best_model(tensor_frame)
            # Apply the softmax function to convert the logits to probabilities.
            current_probabilities = F.softmax(output, dim=1)

    # Update and average the predictions.
    prediction_history.append(current_probabilities)
    if len(prediction_history) > 0:
        # Stack the tensors in the deque and compute the mean along the batch dimension.
        average_probabilities = torch.mean(torch.stack(list(prediction_history)), dim=0)
    else:
        # Fallback for the first few frames.
        average_probabilities = current_probabilities

    # Apply the GUI.
    output_frame = create_GUI(frame, average_probabilities)
    
    # If no face is detected, show an error message.
    if not(len(faces) > 0):
        cv2.putText(output_frame, "No Face is detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame.
    cv2.imshow("PC-Camera", output_frame)

    # End the program if 'q' pressed.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
