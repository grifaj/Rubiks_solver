import cv2 as cv
import numpy as np
from cube import RubiksCube
import time
from importedStateSolve import solve_cube_kociemba
from faceDetector import getState
from showMoves import show_moves
import globals

def checkFront(moves, cube):
    new_moves  = []
    index = 0
    for move in moves:     
        prev =cube.getArray()[0]
        if type(prev) != list:
            prev = prev.tolist()
        cube.move2func(move)
        expected =cube.getArray()[0].tolist()
        if prev == expected and move[0] in ['f', 'b']:
            # add y rotation
            new_moves.append(('y', 'c'))
            f = 'l' if move[0] == 'f' else 'r'
            if move[1] == 'c':
                new_moves.append(('l', 'c'))
            else:
                new_moves.append(('l', 'ac'))
            # recalcuate rest of the moves
            new_moves = new_moves + solve_cube_kociemba(cube)
        else:
            new_moves.append(move)
        index +=1
    
    return new_moves


# main starting function
def run(frame):
    global state
    global moves
    global solved

    # get state from frame
    if state is None:
        state = getState(frame)
        globals.state = state

    elif moves is None:
        # create cube object
        cube = RubiksCube(state=state)
        cube.printCube()

        # get moves from state solve
        print('loading solved')
        #moves = solve_cube(cube)
        moves = solve_cube_kociemba(cube)

        # add rotation if front move is unchanging
        #moves = checkFront(moves, cube)

        globals.moves = moves
        print(globals.moves)

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