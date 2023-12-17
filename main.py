import cv2
import numpy as np

# Load the pre-trained eye cascade classifier
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Load the video file
video_capture = cv2.VideoCapture('IrisPupil.mp4')

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    if not ret:
        break  # Break the loop if the video ends

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the frame
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (ex, ey, ew, eh) in eyes:
        # Draw bounding boxes around the detected eyes
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Eye Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
