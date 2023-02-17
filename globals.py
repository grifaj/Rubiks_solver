from cubeMini import RubiksCube

def init():
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
    global heuristic
    heuristic = None

    # globals for show moves
    global moves
    moves = None
    global state
    state = None
    global moveCount
    moveCount = 0