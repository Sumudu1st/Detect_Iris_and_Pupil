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

    # Iterate over the detected eyes
    for (ex, ey, ew, eh) in eyes:
        # Get the region of interest (ROI) for eyes within the frame
        roi_gray = gray[ey:ey + eh, ex:ex + ew]

        # Apply GaussianBlur to reduce noise
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

        # Use HoughCircles to detect circles (iris and pupils)
        circles = cv2.HoughCircles(
            roi_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=10, maxRadius=30
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Sort circles based on x-coordinate
            circles = sorted(circles[0, :], key=lambda x: x[0])

            # Draw bounding boxes for irises and pupils
            for i, (x, y, r) in enumerate(circles):
                x = ex + x - r
                y = ey + y - r
                w = h = 2 * r

                # Draw bounding box
                color = (0, 255, 0) if i < 2 else (0, 0, 255)  # Green for irises, red for pupils
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Display the frame
    cv2.imshow('Iris and Pupil Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()
