# import the necessary packages
from centroidtracker import CentroidTracker
from faceDetector import getFace
import numpy as np
import cv2 as cv
import time


def run(frame):
	#get centres
	result = getFace(frame, verbose=False)
	if result is not None:
		[centroids, face] = result
	else:
		return
	
	# box rectangles
	objects = ct.update(centroids)
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)



# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
frame_rate = 20 # set frame rate
prev = 0
cap = cv.VideoCapture(0)
if not cap.isOpened():
	print("Cannot open camera")
	exit()

index = 0
prev_colour_time = 0
while True:
	time_elapsed = time.time() - prev
	ret, frame = cap.read()

	colour_time_elapsed = time.time() - prev_colour_time

	if not ret:
		print("Can't receive frame (stream end?). Exiting ...")
		break

	if time_elapsed > 1./frame_rate:
		prev = time.time()

		## per frame operations ##
		run(frame)

		# Display the resulting frame
		cv.imshow('frame',frame)

	if cv.waitKey(1) == ord('q'):
		break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
f.close()

cv.waitKey(0)
