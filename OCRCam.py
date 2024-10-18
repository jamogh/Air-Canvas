import pytesseract
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize drawing points and indices
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Initialize color index and points for drawing
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# Default to blue color
colorIndex = 0  # 0 for blue, 1 for green, 2 for red, 3 for yellow

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

# Initialize the canvas
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 3)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 3)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 3)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 3)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 3)

# Adjusted font sizes and thickness for better visibility
cv2.putText(paintWindow, "CLEAR", (49, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (278, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (408, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (510, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

# Initialize Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to preprocess image for OCR
def preprocess_image_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    return binary_image

# Function to perform OCR using pytesseract
def perform_ocr(image):
    preprocessed_image = preprocess_image_for_ocr(image)
    pil_image = Image.fromarray(preprocessed_image)
    text = pytesseract.image_to_string(pil_image)
    return text

# Main loop
ret = True
while ret:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Drawing rectangles for UI with adjusted thickness and contrast
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 3)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 3)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 3)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 3)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 3)

    # Text with larger font and thickness
    cv2.putText(frame, "CLEAR", (49, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (278, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "RED", (408, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (510, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

    # Capture hand landmarks
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])

        if (thumb[1] - center[1] < 30):
            # New stroke; append points for all colors
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                blue_index = green_index = red_index = yellow_index = 0
                paintWindow[67:, :, :] = 255

            elif 160 <= center[0] <= 255:  # Blue Color
                colorIndex = 0
            elif 275 <= center[0] <= 370:  # Green Color
                colorIndex = 1
            elif 390 <= center[0] <= 485:  # Red Color
                colorIndex = 2
            elif 505 <= center[0] <= 600:  # Yellow Color
                colorIndex = 3

        else:
            # Append to current deque
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)

    # Draw lines on canvas
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 3)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 3)

    # Display the frame and paint window
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    # If the user presses 's', perform OCR
    if cv2.waitKey(1) == ord('s'):
        cropped_image = paintWindow[67:, :, :]  # Capture drawn region
        recognized_text = perform_ocr(cropped_image)

        # Create a new window for displaying recognized text
        text_window = np.zeros((300, 600, 3)) + 255  # White background for the new window
        cv2.putText(text_window, "Recognized Text:", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(text_window, recognized_text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("OCR Result", text_window)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
