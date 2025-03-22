import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
import pyautogui
import os

# Path to the gesture recognition model
model_path = "CSCI376-DS2-main/gesture_recognizer.task"  # Update this to the correct path where the model is saved
if os.path.exists(model_path):
    print("Model file found")
else:
    print("Model file NOT found")

# Initialize the Gesture Recognizer
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1
)
gesture_recognizer = GestureRecognizer.create_from_options(options)

# Initialize Mediapipe hands for custom pointing gesture detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to detect if the index finger is definitively pointing right or left
def recognize_pointing(hand_landmarks, confidence, horizontal_threshold = 0.3, thumb_threshold = 0.1):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]  # Index finger base
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]  # Thumb base joint

    # Check if the index finger is definitively angled to the side
    index_vector = (index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
    finger_slope = index_vector[1] / index_vector[0] if index_vector[0] != 0 else float('inf')  # Slope of the finger
    if abs(finger_slope) < horizontal_threshold:  # Ensure the finger is roughly horizontal (adjust threshold as needed)
        if thumb_tip.y - thumb_base.y > thumb_threshold or thumb_base.y - thumb_tip.y > thumb_threshold:
            if thumb_tip.y < thumb_base.y:
                print("Thumbs Up Detected")
            elif thumb_tip.y > thumb_base.y:
                print("Thumbs Down Detected")

        if index_tip.x > wrist.x:
            print("Pointing Right")
            return "Pointing Right"
        elif index_tip.x < wrist.x:
            print("Pointing Left")
            return "Pointing Left"

    if index_tip.y < wrist.y:  # If the index tip is above the wrist, the finger is pointing up
        if ((thumb_tip.x < index_tip.x or thumb_tip.x > index_tip.x)
                and (thumb_tip.x - index_tip.x > thumb_threshold or index_tip.x - thumb_tip.x > thumb_threshold)):
            print("Thumbs Up Detected")
        return "Pointing Up"
    elif index_tip.y > wrist.y:  # If the tip is below the wrist, the finger is pointing down
        if ((thumb_tip.x < index_tip.x or thumb_tip.x > index_tip.x)
                and (thumb_tip.x - index_tip.x > thumb_threshold or index_tip.x - thumb_tip.x > thumb_threshold)):
            print("Thumbs Up Detected")
        return "Pointing Down"
    return None