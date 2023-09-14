import cv2

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video capture device (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply Gaussian blur to the detected face(s) and keep the rest of the frame as it is
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 0)
        frame[y:y+h, x:x+w] = blurred_face

    # Display the frame with blurred face(s)
    cv2.imshow('Blurred Faces', frame)

    # Break the loop when the Enter key is pressed
    if cv2.waitKey(1) == 13:  # 13 is the ASCII code for Enter
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
