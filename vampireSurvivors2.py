import numpy as np
import mediapipe as mp
import cv2
import time
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def is_palm_open(hand_landmarks):
    # Landmarks for fingertips and respective PIP joints
    fingertip_indices = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                         mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                         mp_hands.HandLandmark.PINKY_TIP]
    pip_indices = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                   mp_hands.HandLandmark.PINKY_PIP]

    open_fingers = 0

    for fingertip, pip in zip(fingertip_indices, pip_indices):
        if hand_landmarks.landmark[fingertip].y < hand_landmarks.landmark[pip].y:
            open_fingers += 1

    return open_fingers >= 4  # Consider palm open if at least 4 fingers are open

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4) as hands:


    last_pressed_key = None

    pressed_keys = set()
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        h, w, c = image.shape
        grid_width, grid_height = w // 3, h // 3

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for i in range(1, 3):
            cv2.line(image, (grid_width * i, 0), (grid_width * i, h), (255, 255, 255), 2)
            cv2.line(image, (0, grid_height * i), (w, grid_height * i), (255, 255, 255), 2)

        current_input = "None"  # Default input
        current_keys = set()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                grid_x, grid_y = x // grid_width, y // grid_height

                palm_open = is_palm_open(hand_landmarks)

                if grid_x == 1 and grid_y == 1 and palm_open:
                    current_input = "Enter"
                    current_keys.add('enter')
                elif grid_x == 0 and grid_y == 0:
                    current_input = "Up & Left"
                    current_keys.update(['up', 'left'])
                elif grid_x == 1 and grid_y == 0:
                    current_input = "Up"
                    current_keys.add('up')
                elif grid_x == 2 and grid_y == 0:
                    current_input = "Up & Right"
                    current_keys.update(['up', 'right'])
                elif grid_x == 0 and grid_y == 1:
                    current_input = "Left"
                    current_keys.add('left')
                elif grid_x == 2 and grid_y == 1:
                    current_input = "Right"
                    current_keys.add('right')
                elif grid_x == 0 and grid_y == 2:
                    current_input = "Down & Left"
                    current_keys.update(['down', 'left'])
                elif grid_x == 1 and grid_y == 2:
                    current_input = "Down"
                    current_keys.add('down')
                elif grid_x == 2 and grid_y == 2:
                    current_input = "Down & Right"
                    current_keys.update(['down', 'right'])
                

                # if grid_x == 1 and grid_y == 1 and palm_open:
                #     current_input = "Enter"
                #     if last_pressed_key != 'enter':
                #         if last_pressed_key is not None:
                #             pyautogui.keyUp(last_pressed_key)
                #         pyautogui.keyDown('enter')
                #         last_pressed_key = 'enter'
                # else:
                #     if last_pressed_key is not None:
                #         pyautogui.keyUp(last_pressed_key)
                #         last_pressed_key = None

                #     if grid_x == 0 and grid_y == 0:
                #         current_input = "Up & Left"
                #         pyautogui.keyDown('up')
                #         pyautogui.keyDown('left')
                #     elif grid_x == 1 and grid_y == 0:
                #         current_input = "Up"
                #         pyautogui.keyDown('up')
                #     elif grid_x == 2 and grid_y == 0:  # Up and Right
                #         current_input = "Up & Right"
                #         pyautogui.keyDown('up')
                #         pyautogui.keyDown('right')
                #     elif grid_x == 0 and grid_y == 1:  # Left
                #         current_input = "Left"
                #         pyautogui.keyDown('left')
                #     elif grid_x == 2 and grid_y == 1:  # Right
                #         current_input = "Right"
                #         pyautogui.keyDown('right')
                #     elif grid_x == 0 and grid_y == 2:  # Left and Down
                #         current_input = "Left & Down"
                #         pyautogui.keyDown('down')
                #         pyautogui.keyDown('left')
                #     elif grid_x == 1 and grid_y == 2:  # Down
                #         current_input = "Down"
                #         pyautogui.keyDown('down')
                #     elif grid_x == 2 and grid_y == 2:  # Right and Down
                #         current_input = "Right & Down"
                #         pyautogui.keyDown('right')
                #         pyautogui.keyDown('')


                # if grid_x == 1 and grid_y == 1 and palm_open:  # Center grid and open palm
                #     current_input = "Enter"
                #     pyautogui.press('enter')
                # else:
                #     if grid_x == 0 and grid_y == 0:  # Up and Left
                #         current_input = "Up & Left"
                #         pyautogui.press(['up', 'left'])
                #     elif grid_x == 1 and grid_y == 0:  # Up
                #         current_input = "Up"
                #         pyautogui.press('up')
                #     elif grid_x == 2 and grid_y == 0:  # Up and Right
                #         current_input = "Up & Right"
                #         pyautogui.press(['up', 'right'])
                #     elif grid_x == 0 and grid_y == 1:  # Left
                #         current_input = "Left"
                #         pyautogui.press('left')
                #     elif grid_x == 2 and grid_y == 1:  # Right
                #         current_input = "Right"
                #         pyautogui.press('right')
                #     elif grid_x == 0 and grid_y == 2:  # Left and Down
                #         current_input = "Left & Down"
                #         pyautogui.press(['left', 'down'])
                #     elif grid_x == 1 and grid_y == 2:  # Down
                #         current_input = "Down"
                #         pyautogui.press('down')
                #     elif grid_x == 2 and grid_y == 2:  # Right and Down
                #         current_input = "Right & Down"
                #         pyautogui.press(['right', 'down'])

        # Process key presses and releases
        keys_to_release = pressed_keys - current_keys
        keys_to_press = current_keys - pressed_keys

        for key in keys_to_release:
            pyautogui.keyUp(key)
        for key in keys_to_press:
            pyautogui.keyDown(key)

        pressed_keys = current_keys.copy()       

        cv2.putText(image, f'Input: {current_input}', (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
