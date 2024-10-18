import easyocr
import cv2
import matplotlib.pyplot as plt

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], verbose=False)

# Load the image
image_path = r"E:\AMJ PROJECTS\Air-Canvas\captured_paint.png"
image = cv2.imread(image_path)

# Perform OCR on the image
results = reader.readtext(image)

# Display the results
for (bbox, text, prob) in results:
    # Draw the bounding box on the image
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Display the recognized text
    print(f"Detected text: {text} (Confidence: {prob:.2f})")

# Show the image with bounding boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
