from cubeMini import RubiksCube
import json

def init(saveHeuristic=False):
    print('being used')
    # globals for face detector
    global update_colours
    update_colours  = False
    global faceNum
    faceNum = 0
    global consistentCount
    consistentCount = 0
    global previousFace
    previousFace = None
    global rotateFlag
    rotateFlag = False
    global lastScan
    lastScan = []
    global detectedCube
    detectedCube = RubiksCube(state='xxxxxxxxxxxxxxxxxxxxxxxx')

    # globals for state solve
    if not saveHeuristic:
        global heuristic
        path = '/home/grifaj/Documents/y3project/Rubiks_solver/'
        with open(path+'heuristic.json') as f:
            heuristic = json.load(f)

    # globals for show moves
    global moves
    moves = None
    global state
    state = None
    global moveCount
    moveCount = 0

    if __name__ == '__main__':
        init()