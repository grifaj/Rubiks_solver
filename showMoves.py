import cv2 as cv
import numpy as np
from cubeMini import RubiksCube
import time
from stateSolve import CubeSolver

def show_moves(frame, moves, cube):
    # draw arrows on cube to show moves
    move = moves[moveCount]

    if not moveMade:    
        if move[0] == 'f':
            # draw arrow on top 2 cubes
            cv.arrowedLine(frame,(100,100),(200,200), (0,255,0),9)

    # check if front face is what is expected then show next move
    #face = get_face(frame)
    face = None
    expected = cube.move2func(move)[0] # front face it 0 I think?
    if face == expected:
        moveMade = True
        moveCount +=1

    if moveCount == len(moves):
        print('cube solved')

    return frame

# main starting function
def run(frame):
    global solver
    global moves
    global state

    if state is None:
        # get state from frame
        #state = get_state(frame)
        state = 'bobyggrwwbrywwygyorgrobo' # hard code state for now

    cube = RubiksCube(state=state)

    if moves is None:
        # get moves from state solve
        #cost, moves = solver.solve_cube(cube)
        moves = [('f', 'c'), ('u', 'c'), ('l', 'c')]

    # output moves on the cube
    output = show_moves(frame, moves,cube)

    return output

# pre-capture setup
#solver = CubeSolver()
moves = None
state = None
moveCount = 0
moveMade = False

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
        output = run(frame)

        # Display the resulting frame
        cv.imshow('frame',output)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

cv.waitKey(0)