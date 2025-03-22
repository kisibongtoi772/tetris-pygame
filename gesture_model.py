import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
import pyautogui
import os

# Path to the gesture recognition model
model_path = "gestureModel/CSCI376-DS2-main/gesture_recognizer.task"  # Update this to the correct path where the model is saved
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
def recognize_pointing(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]  # Index finger base
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Check if the index finger is definitively angled to the side
    index_vector = (index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
    finger_slope = index_vector[1] / index_vector[0] if index_vector[0] != 0 else float('inf')  # Slope of the finger
    print(abs(finger_slope))
    if abs(finger_slope) < 0.3:  # Ensure the finger is roughly horizontal (adjust threshold as needed)
        if index_tip.x > wrist.x:
            print("MOVE_RIGHT")
            return "Pointing Right"
        elif index_tip.x < wrist.x:
            print("MOVE_LEFT")
            return "Pointing Left"
        print(abs(finger_slope))
        print(wrist.x)
    # if abs(finger_slope) > 0.3:  # Ensure the finger is roughly horizontal (adjust threshold as needed)
    #     if abs(finger_slope) > 5:
    #         print("MOVE_UP")
    #         return "Pointing Up"
    #     elif abs(finger_slope) < 5:
    #         print("MOVE_DOWN")
    #         return "Pointing Down"
    #     print(abs(finger_slope))

    # Check if the finger is pointing upwards or downwards (based on the y-coordinate)
    if index_tip.y < wrist.y:  # If the tip is above the wrist, the finger is pointing up
        print("MOVE_UP")
        return "Pointing Up"
    elif index_tip.y > wrist.y:  # If the tip is below the wrist, the finger is pointing down
        print("MOVE_DOWN")
        return "Pointing Down"
    return None
