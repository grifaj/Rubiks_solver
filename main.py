import cv2 as cv
import numpy as np
from cubeMini import RubiksCube
import time
from stateSolve import solve_cube
from faceDetector import getState
from showMoves import show_moves
import globals

# main starting function
def run(frame):
    global state
    global moves
    global solved

    # get state from frame
    if state is None:
        state = getState(frame)
        '''tmp = RubiksCube()
        tmp.right_prime()
        state = tmp.stringify()'''
        globals.state = state

    elif moves is None:
        # create cube object
        cube = RubiksCube(state=state)
        cube.printCube()

        # get moves from state solve
        print('loading solved')
        moves = solve_cube(cube)
        globals.moves = moves
        print(moves)

    elif not solved:
        # output moves on the cube
        solved = show_moves(frame)

state = None
moves = None
solved = False

# intit globals
globals.init()

frame_rate = 20 # set frame rate
prev =0
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
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