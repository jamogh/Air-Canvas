import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and drawing utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Set the width and height of the output screen
frameWidth = 640
frameHeight = 480

# Capture video from the webcam
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# Color value for drawing (BGR format)
drawColor = (255, 0, 0)

# Create a transparent canvas
canvas = np.zeros((frameHeight, frameWidth, 4), dtype=np.uint8)

# Initialize myPoints to store points [x, y, mode]
myPoints = []

# Function to draw and erase on the canvas
def drawAndEraseOnCanvas(myPoints, drawColor, canvas):
    for point in myPoints:
        if point[2] == 'draw':
            cv2.circle(canvas, point[:2], 10, drawColor + (255,), cv2.FILLED)
        elif point[2] == 'erase':
            cv2.circle(canvas, point[:2], 50, (0, 0, 0, 0), cv2.FILLED)

# Running an infinite loop so that the program keeps running until we close it
while True:
    try:
        success, img = cap.read()
        if not success:
            break

        # No flipping here for processing, but we'll flip for display later
        imgResult = img.copy()

        # Convert the BGR image to RGB before processing
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, c = img.shape

                # Get the index fingertip position (landmark 8)
                finger_tip = handLms.landmark[8]
                cx, cy = int(finger_tip.x * w), int(finger_tip.y * h)

                # Get the bounding box of the hand to determine if the full palm is visible
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in handLms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                # Calculate the area of the bounding box
                bbox_area = (x_max - x_min) * (y_max - y_min)

                # Erase if the bounding box is large enough to represent the full palm
                if bbox_area > 20000:  # Adjust the threshold according to your requirements
                    myPoints.append((cx, cy, 'erase'))
                else:
                    # Append the position to the list of points for drawing
                    myPoints.append((cx, cy, 'draw'))

                # Draw the landmarks and connections on the hand
                mpDraw.draw_landmarks(imgResult, handLms, mpHands.HAND_CONNECTIONS)

        # Draw or erase on the canvas
        if len(myPoints) != 0:
            drawAndEraseOnCanvas(myPoints, drawColor, canvas)

        # Combine the canvas with the webcam feed
        imgResult = cv2.addWeighted(imgResult, 1, canvas[:, :, :3], 1, 0)

        # Flip the final result for display
        imgResult = cv2.flip(imgResult, 1)

        # Displaying output on screen
        cv2.imshow("Result", imgResult)

        # Condition to break the execution of the program
        # Press 'q' to stop the execution
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()