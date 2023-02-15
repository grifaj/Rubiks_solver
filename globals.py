def init():
    # globals for face detector
    global update_colours
    update_colours  = False
    global array
    array = []
    global faceNum
    faceNum = 0
    global consistentCount
    consistentCount = 0
    global previousFace
    previousFace = None

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