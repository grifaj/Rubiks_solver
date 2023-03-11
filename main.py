import cv2 as cv
import numpy as np
from cubeMini import RubiksCube
import time
from stateSolve import solve_cube, colour_independant
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
        # check state is possible
        if state is not None and not colour_independant(state) in globals.heuristic:
            print('impossible state, redoing scan')
            state = None
            globals.init(saveHeuristic=True)
        # check state is possible by counting colours
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
    
    else:
        cv.putText(img=frame, text='Cube solved', org=(100, 150), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 255, 0),thickness=2)


state = None
moves = None
solved = False
record = False

# intit globals
globals.init()

frame_rate = 20 # set frame rate
prev =0
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

# setting up video recording 
if record:
    width= int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer= cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (width,height))

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

        if record:
            # save frame
            writer.write(frame)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

cv.waitKey(0)