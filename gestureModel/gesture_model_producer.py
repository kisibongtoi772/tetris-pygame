import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
from mediapipe.tasks.python import BaseOptions
import pyautogui
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.kafka_producer import TetrisKafkaProducer
import time

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
def recognize_pointing(hand_landmarks, confidence, horizontal_threshold=0.3, thumb_threshold=0.1, vertical_threshold=3.0):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]  # Index finger base
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]  # Index middle joint
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]  # Thumb base joint

    # Check if the index finger is definitively angled to the side
    index_vector = (index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
    finger_slope = index_vector[1] / index_vector[0] if index_vector[0] != 0 else float('inf')  # Slope of the finger
    
    # Calculate the absolute slope for verticality check
    abs_slope = abs(finger_slope)
    
    if abs_slope < horizontal_threshold:  # Ensure the finger is roughly horizontal
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

    # Check for index finger pointing up vs thumb up
    if index_tip.y < wrist.y:  # We're in the upper region
        # Check if fingers are curled (except thumb)
        index_curled = index_tip.y > index_pip.y 
        middle_curled = middle_tip.y > middle_pip.y
        
        # Check if thumb is extended upward significantly
        thumb_extended_up = thumb_tip.y < thumb_base.y - 0.12
        
        # If thumb is up and fingers are curled, it's a thumb up gesture
        if thumb_extended_up and index_curled and middle_curled:
            print("Thumb Up - Confirmed")
            return "Thumb_Up"
            
        # For pointing up, require the finger to be nearly vertical
        if not index_curled:
            # Check if the index finger is vertical enough
            # abs_slope is large when vertical, small when horizontal
            print(f"Finger slope: {abs_slope:.2f}, threshold: {vertical_threshold}")
            
            if abs_slope > vertical_threshold:  # Requiring a steep slope for vertical pointing
                # Also ensure the finger is actually pointing up not down
                if index_tip.y < index_mcp.y:  # Tip is above base
                    print(f"Pointing Up - Verified Vertical (slope: {abs_slope:.2f})")
                    return "Pointing Up"
                else:
                    print("Not pointing Up - finger tip below base")
            else:
                print(f"Not vertical enough for Pointing Up (slope: {abs_slope:.2f})")
            
    elif index_tip.y > wrist.y:  # If the tip is below the wrist, the finger is pointing down
        if ((thumb_tip.x < index_tip.x or thumb_tip.x > index_tip.x)
                and (thumb_tip.x - index_tip.x > thumb_threshold or index_tip.x - thumb_tip.x > thumb_threshold)):
            print("Thumbs Down Detected")
            return "Thumb_Down"
        return "Pointing Down"
            
    return None

last_command_time = 0
COMMAND_COOLDOWN = 0.5
def main():
    global last_command_time, last_speed_time
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    kafka_producer = TetrisKafkaProducer().get_instance()
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            current_time = time.time()
            can_send_command = current_time - last_command_time > COMMAND_COOLDOWN
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally and convert the BGR image to RGB.
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to a Mediapipe Image object for the gesture recognizer
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Perform gesture recognition on the image
            result = gesture_recognizer.recognize(mp_image)

            # Convert the image to hand landmarks to detect pointing gestures
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if result.gestures:
                printed_gesture = None
                recognized_gesture = result.gestures[0][0].category_name
                confidence = result.gestures[0][0].score

            
                # Recognize pointing gestures with hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        
                        pointing_direction = recognize_pointing(hand_landmarks, confidence)
                        if can_send_command:
                            print(f"Pointing Direction: {pointing_direction}")
                            if pointing_direction == "Pointing Right":
                                # pyautogui.press("right")
                                kafka_producer.send_command("right")
                                last_command_time = current_time

                            if pointing_direction == "Pointing Left":
                                # pyautogui.press("left")
                                kafka_producer.send_command("left")
                                last_command_time = current_time

                            if pointing_direction == "Pointing Up":
                                # pyautogui.press("up")
                                kafka_producer.send_command("speed")
                                last_command_time = current_time

                            elif pointing_direction == "Pointing Down":
                                kafka_producer.send_command("down")
                                last_command_time = current_time
                            
                                # pyautogui.press("down")
                            
                            
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        printed_gesture = pointing_direction

                # Example of pressing keys with pyautogui based on other recognized gestures
                if recognized_gesture == "Open_Palm":
                    printed_gesture = recognized_gesture
                    #print("pause")
                    kafka_producer.send_command("pause")
                elif recognized_gesture == "Thumb_Up":
                    printed_gesture = recognized_gesture
                    #print("thumb down")
                    kafka_producer.send_command("yes")
                elif recognized_gesture == "Thumb_Down":
                    printed_gesture = recognized_gesture
                    #print("thumb down")
                    kafka_producer.send_command("no")
                elif recognized_gesture == "Victory":
                    printed_gesture = recognized_gesture
                    #print("victory")
                    pyautogui.press("space")

                pyautogui.PAUSE=0.01

                # Display recognized gesture and confidence 
                cv2.putText(image, f"Gesture: {printed_gesture} ({confidence:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


            # Display the resulting image (can comment this out for better performance later on)
            cv2.imshow('Gesture Recognition', image)

            if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to quit
                break

    cap.release()
    cv2.destroyAllWindows()
    kafka_producer.close()


if __name__ == "__main__":
    main()
