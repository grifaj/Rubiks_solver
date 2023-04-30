import cv2 as cv
import numpy as np
import time

def run(frame):
    # Load image and convert to grayscale
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Perform edge detection using Canny edge detector
    blur = cv.blur(frame,(3,3))
    canny = cv.Canny(blur, 50, 100, L2gradient = True) #60, 100
    #edges = cv.Canny(gray, 50, 150, apertureSize=3)

    # Perform Hough transform to detect lines in the image
    lines = cv.HoughLines(canny, rho=1, theta=np.pi/180, threshold=115)

    # Draw the lines on the original image
    try:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    except:
        pass
    # Display the resulting image with lines detected
    #cv.imshow('Hough Transform', frame)


frame_rate = 20 # set frame rate
prev =0
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
#cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

while True:
    time_elapsed = time.time() - prev
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if time_elapsed > 1./frame_rate:
        prev = time.time()

        ## per frame operations ##
        frame = cv.flip(frame, 1)
        run(frame)

        # Display the resulting frame
        cv.imshow('frame',frame)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

cv.waitKey(0)
