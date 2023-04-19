from cubeMini import RubiksCube
import h5py

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
        # Open the HDF5 file for reading
        h = h5py.File('heuristic.hdf5', 'r')
        data = h['data'][:]
        heuristic = {row[0].decode('utf-8'): row[1] for row in data}
        h.close()

    # globals for show moves
    global moves
    moves = None
    global state
    state = None
    global moveCount
    moveCount = 0

    if __name__ == '__main__':
        init()