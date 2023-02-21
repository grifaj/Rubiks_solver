import cv2 as cv
import numpy as np
import joblib

class Cubie:

    def __init__(
        self,
        frame,
        contour,
        colour = None,
        hsvVal = None,
        centre = None
    ):

        self.frame = frame
        self.contour = contour

        if colour is None:
            self.setColour()

        if centre is None:
            self.setCentre()

    def setColour(self):
        # create mask for contour
        grey = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros(grey.shape, np.uint8) 
        cv.drawContours(mask, [self.contour], 0, 255, -1)

        # convert to lab colour space
        lab_img = cv.cvtColor(self.frame, cv.COLOR_BGR2LAB)

        # calculate mean pixel colour in contour
        mean = cv.mean(lab_img, mask=mask)[1:-1]
        hsv = [int(a) for a in mean] # pure colour
        mean = np.uint8([[mean]])[0][0]
        mean = mean.tolist()

        #try knn for classifiing colours
        knn = joblib.load('knn.joblib')
        colour_list = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]
        index = knn.predict([mean])[0]

        self.colour = colour_list[index]
        self.hsvVal = hsv

    def setCentre(self):
        M = cv.moments(self.contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        self.centre = (cX, cY)