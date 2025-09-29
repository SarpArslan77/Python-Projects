import cv2
import numpy as np
import time

# --- Configuration and Constants ---
# A list of the 7 emotions your model can predict
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# A dictionary to map emotions to colors for Idea 2
EMOTION_COLORS = {
    "Angry":    (0, 0, 255),     # Red
    "Disgust":  (0, 128, 0),     # Green
    "Fear":     (128, 0, 128),   # Purple
    "Happy":    (0, 255, 255),   # Yellow
    "Sad":      (255, 0, 0),     # Blue
    "Surprise": (255, 165, 0),   # Orange
    "Neutral":  (128, 128, 128)  # Gray
}

# --- SIMULATION HELPER FUNCTIONS (No need to change) ---

def simulate_model_predictions():
    """Generates a list of 7 random floats that sum to 1.0, like a softmax output."""
    raw_predictions = np.random.rand(7)
    return raw_predictions / np.sum(raw_predictions)

def get_simulated_face_box():
    """Returns a static bounding box to simulate face detection."""
    # Format: (x_start, y_start, x_end, y_end)
    return (170, 90, 470, 390)

# --- GUI DRAWING FUNCTIONS ---

def draw_idea_1_bar_chart(frame, predictions):
    """Draws a dynamic bar chart on the right side of the frame."""
    canvas = frame.copy()
    # Find the winning emotion
    top_emotion_index = np.argmax(predictions)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
        # Calculate bar properties
        bar_x = 10
        bar_y = i * 40 + 50
        bar_width = int(prob * 200)  # Scale probability to pixel width
        
        # Highlight the top emotion's bar and text
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        if i == top_emotion_index:
            color = (0, 255, 0) # Bright Green for the winner

        # Draw text (Emotion: XX.X%)
        text = f"{emotion}: {prob:.1%}"
        cv2.putText(canvas, text, (bar_x, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw the probability bar
        cv2.rectangle(canvas, (bar_x + 180, bar_y + 5), (bar_x + 180 + bar_width, bar_y + 25), color, -1)
        
    return canvas


def draw_idea_2_bounding_box(frame, predictions):
    """Draws a color-coded bounding box and text label around the 'face'."""
    canvas = frame.copy()
    x1, y1, x2, y2 = get_simulated_face_box()

    # Find the winning emotion and its probability
    top_emotion_index = np.argmax(predictions)
    top_emotion = EMOTIONS[top_emotion_index]
    top_prob = predictions[top_emotion_index]
    
    # Get the corresponding color
    box_color = EMOTION_COLORS.get(top_emotion, (255, 255, 255)) # Default to white
    
    # Draw the bounding box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 3)

    # Create the text label (e.g., "Happy: 92%")
    label = f"{top_emotion}: {top_prob:.0%}"
    
    # Draw a filled background for the text for better readability
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(canvas, (x1, y1 - h - 15), (x1 + w, y1), box_color, -1)
    
    # Draw the text on the background
    cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return canvas


def create_idea_3_dashboard(frame, predictions):
    """Creates a separate dashboard panel and stitches it to the camera feed."""
    # Create a black canvas for the dashboard (height must match frame)
    dashboard_width = 320
    dashboard = np.zeros((frame.shape[0], dashboard_width, 3), dtype="uint8")

    # --- Draw elements on the dashboard ---
    cv2.putText(dashboard, "Emotion Analysis", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    top_emotion_index = np.argmax(predictions)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
        bar_y = i * 40 + 80
        
        # Highlight the top emotion
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))
        if i == top_emotion_index:
            color = (0, 255, 0)
        
        # Format text and draw it
        text = f"{emotion}: {prob:.1%}"
        cv2.putText(dashboard, text, (10, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw the bar
        bar_width = int(prob * (dashboard_width - 130)) # Adjust bar scale for dashboard width
        cv2.rectangle(dashboard, (180, bar_y + 5), (180 + bar_width, bar_y + 25), color, -1)
    
    # Stitch the camera frame and the dashboard together horizontally
    combined_frame = np.hstack([frame, dashboard])
    return combined_frame


# --- MAIN FUNCTION ---

def main():
    """Main loop to capture video and display the selected GUI."""
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! CHANGE THIS VALUE TO 1, 2, or 3 TO SEE THE DIFFERENT IDEAS !!!
    selected_idea = 2
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a mirror-like view
        frame = cv2.flip(frame, 1)

        # Get simulated predictions for this frame
        predictions = simulate_model_predictions()

        # Apply the selected GUI idea
        if selected_idea == 1:
            output_frame = draw_idea_1_bar_chart(frame, predictions)
            window_title = "Idea 1: Bar Chart Overlay"
        elif selected_idea == 2:
            output_frame = draw_idea_2_bounding_box(frame, predictions)
            window_title = "Idea 2: Color-Coded Bounding Box"
        elif selected_idea == 3:
            output_frame = create_idea_3_dashboard(frame, predictions)
            window_title = "Idea 3: Separate Dashboard Panel"
        else:
            output_frame = frame
            window_title = "Original Frame"

        # Display the resulting frame
        cv2.imshow(window_title, output_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()