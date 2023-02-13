import cv2 as cv
import numpy as np
from cubeMini import RubiksCube
import time
from stateSolve import CubeSolver
from faceDetector import getFace

# returns the centre of all the centres
def getCentre(centres):
    return (int(np.mean([ i for i, _ in centres])),int(np.mean([ j for _, j in centres])))

def draw_triangle(img, centre, side_length):
    # Calculate the three vertices of the triangle
    vertex1 = (int(centre[0]), int(centre[1] - side_length / 2))
    vertex2 = (int(centre[0] - side_length / 2), int(centre[1] + side_length / 2))
    vertex3 = (int(centre[0] + side_length / 2), int(centre[1] + side_length / 2))

    # Fill the triangle using fillPoly
    triangle = np.array([vertex1, vertex2, vertex3], np.int32)
    triangle = triangle.reshape((-1, 1, 2))
    cv.fillPoly(img, [triangle], (255,0,255))


# very messy code to draw circular arrows on the face
def draw_arc(frame, centres, dir):
    # Ellipse parameters
    radius = int(np.abs(centres[0][0] - centres[1][0])/1.5)
    centre = getCentre(centres)

    # angles top and bottom
    angles = [[210,330],[30,150]]

    # draw both halves of arc
    cv.ellipse(frame, centre, (radius, radius), 0, angles[0][0], angles[0][1], (255,0,255), 8)
    cv.ellipse(frame, centre, (radius, radius), 0, angles[1][0], angles[1][1], (255,0,255), 8)


    index = 0 if dir == 'c' else 1

    x = int(centre[0] + radius * np.cos(np.deg2rad(angles[0][index])))
    y = int(centre[1] + radius * np.sin(np.deg2rad(angles[0][index])))
    draw_triangle(frame, (x,y), 20)

    x = int(centre[0] + radius * np.cos(np.deg2rad(angles[1][index])))
    y = int(centre[1] + radius * np.sin(np.deg2rad(angles[1][index])))
    draw_triangle(frame, (x,y), 20)


# arrows drawn relfected so move is right for person holding it
# can't do front move dirctly ??
def putArrow(frame, move, centres):
    if move[0] == 'u':
        # draw arrow on top 2 cubes
        start = centres[1]
        end = centres[0]
    if move[0] == 'r':
        # 2 left hand cubes
        start = centres[2]
        end = centres[0]
    if move[0] == 'l':
        # 2 right hand cubes
        start = centres[3]
        end = centres[1]
    if move[0] == 'd':
        # bottom 2 cubes
        start= centres[4]
        end = centres[3]

    if move[0] == 'b':
        # special arrow for back
        draw_arc(frame, centres, move[1])
    if move[0] == 'f':
        # might have to rotate the cube, use 2 moves
        # for now draw arc anf but big F on it
        dir = 'c' if move[1] == 'ac' else 'ac' # swap direction
        draw_arc(frame, centres, dir)
        org = getCentre(centres)
        org = (org[0] - 12, org[1] + 10) #ajust centre to take into acont text size
        cv.putText(img=frame, text='F', org=org, fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=1.5, color=(255, 0, 255),thickness=3)


    # reverse arrow if direction if anti-clockwise
    if move[1] == 'ac':
        temp = start
        start = end
        end = temp

    if move[0] != 'b' and move[0] != 'f':
        cv.arrowedLine(frame, start, end, (255,0,255),6, tipLength = 0.2)


def show_exp_move(frame, colours):
    colour_dict = {'w':(255,255,255), 'r':(20,18,137), 'b':(172,72,13), 'o':(37,85,255), 'g':(76,155,25), 'y':(47,213,254)}

    colours = colours.tolist()
    side_len = 50
    w = frame.shape[1] - side_len*2
    for i in range(len(colours)):
        for j in range(len(colours[0])):
            #cv.rectangle(frame, (width-(2-i)*side_len,j*side_len),(width-(2-i)*side_len,j*side_len), colour_dict[colours[i][j]],-1)
            cv.rectangle(frame, (i*side_len+w,j*side_len),((i+1)*side_len+w,(j+1)*side_len), colour_dict[colours[j][i]],-1)

    return frame

def show_moves(frame, moves):
    global moveCount
    global state
    global solved

    try:
        centres, face = getFace(frame)
    except:
        return
   
    if moveCount == len(moves):
        solved = True
        print('cube solved')
        return

    cube = RubiksCube(state=state)
    move = moves[moveCount]

    # draw move on arrow
    putArrow(frame, move, centres)
        
    # check if front face is what is expected then show next move
    cube.move2func(move)
    show_exp_move(frame, cube.getArray()[1])
    expected =cube.getArray()[1].tolist() 
    if face == expected:
        moveCount +=1
        print('move made',moveCount)
        state = cube.stringify()
        cube.printCube()


# main starting function
def run(frame):
    global solver
    global moves
    global state
    global solved

    if state is None:
        print('set state')
        # get state from frame
        #state = get_state(frame)
        state = 'rwryyowowryrwoyobbgggbgb'
        cube = RubiksCube(state=state)
        #print(cube.shuffle(3))
        cube.printCube()
        state = cube.stringify()
        print(state)

    if moves is None:

        cube = RubiksCube(state=state)
        # get moves from state solve
        moves = [('l', 'c'), ('u', 'c'), ('u', 'c')]
        #cost, moves = solver.solve_cube(cube)
        print(moves)

    if not solved:
        # output moves on the cube
        show_moves(frame, moves)

# pre-capture setup
#solver = CubeSolver()
moves = None
state = None
moveCount = 0
solved = False

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
        run(frame)

        # Display the resulting frame
        cv.imshow('frame',frame)

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

cv.waitKey(0)