import cv2

# Create a VideoCapture object to capture frames from the camera
cap = cv2.VideoCapture(0)

# Loop over frames captured from the camera
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    cv2.imshow('orignal frame', frame)
    
    # Apply binary thresholding to the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Display the resulting frame
    cv2.imshow('Thresholded Frame', thresh)
    
    # Wait for a key press to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
