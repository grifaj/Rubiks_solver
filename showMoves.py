import cv2 as cv
import numpy as np
from cubeMini import RubiksCube
from faceDetector import getFace
from stateSolve import solve_cube
import globals

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
        start = centres[3]
        end = centres[1]
    if move[0] == 'l':
        # 2 right hand cubes
        start = centres[0]
        end = centres[2]
    if move[0] == 'd':
        # bottom 2 cubes
        start= centres[4]
        end = centres[3]
    if move[0] == 'b':
        #shouldn't use back
        print('illegal move')
        return
    if move[0] == 'f':
        dir = 'c' if move[1] == 'ac' else 'ac'
        draw_arc(frame, centres, dir)
    if move[0] == 'y':
        cv.arrowedLine(frame, centres[1], centres[0], (255,0,255),6, tipLength = 0.2)
        cv.arrowedLine(frame, centres[3], centres[2], (255,0,255),6, tipLength = 0.2)

    if move[0] != 'f' and move[0] != 'y':
        # reverse arrow if direction if anti-clockwise
        if move[1] == 'ac':
            temp = start
            start = end
            end = temp
        cv.arrowedLine(frame, start, end, (255,0,255),6, tipLength = 0.2)

def getWrongMove(face, previous):
    direction = ['c', 'ac']
    moves = ['f', 'u', 'd', 'l', 'r', 'x' ,'y', 'z']
    for m in moves:
        for d in direction:
            tmp = RubiksCube(state=previous.stringify())
            tmp.move2func((m,d))
            if tmp.getArray()[0].tolist() == face:
                return (m,d)

    return None

def show_exp_move(frame, colours):
    colour_dict = {'w':(255,255,255), 'r':(20,18,137), 'b':(172,72,13), 'o':(37,85,255), 'g':(76,155,25), 'y':(47,213,254)}

    side_len = 50
    w = frame.shape[1] - side_len*2
    cv.putText(img=frame, text='Expected', org=(w -10, 125), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.7, color=(0, 255, 0),thickness=1)
    cv.putText(img=frame, text='face', org=(w+25, 145), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.7, color=(0, 255, 0),thickness=1)
    for i in range(len(colours)):
        for j in range(len(colours[0])):
            cv.rectangle(frame, (i*side_len+w,j*side_len),((i+1)*side_len+w,(j+1)*side_len), colour_dict[colours[j][i]],-1)

    return frame

def show_moves(frame):

    if globals.moveCount == len(globals.moves):
        print('cube solved')
        return True

    cube = RubiksCube(state=globals.state)
    previous = RubiksCube(state=globals.state) # for orignal state
    move = globals.moves[globals.moveCount]
    
    # check if front face is what is expected then show next move
    cube.move2func(move)
    expected =cube.getArray()[0]
    if type(expected) != list: expected = expected.tolist()
    show_exp_move(frame, expected)

    result = getFace(frame, verbose=False)
    if result is not None:
        [centres, face] = result
    else:
        return False
    
     # draw move on arrow
    putArrow(frame, move, centres)

    if face == expected:
        globals.moveCount +=1
        print('move made',globals.moveCount)
        globals.state = cube.stringify()
        #cube.printCube()
    
    else:
        # generate expected faces of different moves that could have been made and find wrong move
        if face != previous.getArray()[0]:
            wrongMove = getWrongMove(face, previous)
            if wrongMove is not None:
                # regenerate moves based on new state
                print('wrong move made',wrongMove)
                previous.move2func(wrongMove)
                globals.state = previous.stringify()
                globals.moves = solve_cube(previous)
                print(globals.moves)
                globals.moveCount = 0
        

    return False
