import numpy as np

# remove none placeholders from slices
def remove_None(cube):
    for a in cube:
        for b in a:
            for c in b:
                try:
                    c.remove(None)
                except ValueError:
                    pass


def front(cube):
    # make temp of top edge
    temp = np.copy(cube[3, 2, :])
    #move left edge to top edge
    cube[3, 2, :] = np.flip(cube[5, :, 2])
    # move bottom edge to left edge
    cube[5, :, 2] = cube[2, 0, :]
    # move right edge to bottom edge
    cube[2, 0, :] = np.flip(cube[4, :, 0])
    # move top edge to right
    cube[4, :, 0] = temp
    # rotate front face
    cube[0, :, :] = np.rot90(cube[0, :, :], -1)

def front_prime(cube):
    # rotate clockwise three times
    for _ in range(3):
        front(cube)

def right(cube):
    # make temp of top edge
    temp = np.copy(cube[3, :, 2])
    #move front edge to top edge
    cube[3, :, 2] = cube[0, :, 2]
    # move bottom edge to front edge
    cube[0, :, 2] = cube[2, :, 2]
    # move back edge to bottom edge
    cube[2, :, 2] = np.flip(cube[1, :, 0])
    # move top edge to back edge
    cube[1, :, 0] = np.flip(temp)
    # rotate right face
    cube[4, :, :] = np.rot90(cube[4, :, :],-1)

def right_prime(cube):
     # rotate clockwise three times
    for _ in range(3):
        right(cube)

def left(cube):
    # make temp of top edge
    temp = np.copy(cube[3, :, 0])
    # move back edge to top edge
    cube[3, :, 0] = np.flip(cube[1, :, 2])
    # move bottom edge to back edge
    cube[1, :, 2] = np.flip(cube[2, :, 0])
    # move front edge to bottom edge
    cube[2, :, 0] = cube[0, :, 0]
    # move top edge to front edge
    cube[0, :, 0] = temp
    # rotate left face
    cube[5, :, :] = np.rot90(cube[5, :, :],-1)


def left_prime(cube):
    # rotate clockwise three times
    for _ in range(3):
        left(cube)

def up(cube):
    # make temp of front
    temp = np.copy(cube[0, 0, :])
    # move right to front
    cube[0, 0, :] = cube[4, 0, :]
    # move back to right
    cube[4, 0, :] = cube[1, 0, :]
    # move left to back
    cube[1, 0, :] = cube[5, 0, :]
    # move front to left
    cube[5, 0, :] = temp
    # rotate up face
    cube[3, :, :] = np.rot90(cube[3, :, :],-1)

def up_prime(cube):
    # rotate clockwise three times
    for _ in range(3):
        up(cube)

def down(cube):
    # make temp of front
    temp = np.copy(cube[0, 2, :])
    # move left to front
    cube[0, 2, :] = cube[5, 2, :]
    # move back to left
    cube[5, 2, :] = cube[1, 2, :]
    # move right to back
    cube[1, 2, :] = cube[4, 2, :]
    # move front to right
    cube[4, 2, :] = temp
    # rotate down face
    cube[2, :, :] = np.rot90(cube[2, :, :],-1)


def down_prime(cube):
    # rotate clockwise three times
    for _ in range(3):
        down(cube)

def back(cube):
    # make temp of top
    temp = np.copy(cube[3, 0, :])
    # move right to top
    cube[3, 0, :] = cube[4, :, 2]
    # move bottom to right
    cube[4, :, 2] = np.flip(cube[2, 2, :])
    # move left to bottom
    cube[2, 2, :] = cube[5, :, 0]
    # move top to left
    cube[5, :, 0] = np.flip(temp)
    # rotate back face
    cube[1, :, :] = np.rot90(cube[1, :, :],-1)


def back_prime(cube):
    # rotate clockwise three times
    for _ in range(3):
        back(cube)


#convert from faces to slices
def face2pieces(cube):
    slices = [[[[] for _ in range(3)] for _ in range(3)] for _ in range(3)]
    for i in range(3):
        slices[0][0][i] = [cube[3,2,i],(cube[5,0,2] if i == 0 else (cube[4,0,0] if i == 2 else None)),cube[0,0,i]] # top row first slice
        slices[0][1][i] = [(cube[5,1,2] if i == 0 else (cube[4,1,0] if i == 2 else None)),cube[0,1,i]] # middle row first slice
        slices[0][2][i] =  [cube[2,0,i], (cube[5,2,2] if i ==0 else (cube[4,2,0] if i == 2 else None)),cube[0,2,i]] # bottom row first slice

        slices[1][0][i] = [cube[3,1,i],(cube[5,0,1] if i == 0 else (cube[4,0,1] if i == 2 else None))] # top row middle slice
        slices[1][2][i] = [cube[2,1,i], (cube[5,2,1] if i == 0 else (cube[4,2,1] if i == 2 else None))] #bottom row middle slice

        slices[2][0][i] = [cube[3,0,i], (cube[5,0,0] if i == 0 else (cube[4,0,2] if i == 2 else None)), cube[1,0,2-i]] # top row back slice
        slices[2][1][i] = [(cube[5,1,0] if i == 0 else (cube[4,1,2] if i == 2 else None)), cube[1,1,2-i]] # middle row back slice
        slices[2][2][i] = [cube[2,2,i], (cube[5,2,0] if i == 0 else (cube[4,2,2] if i == 2 else None)), cube[1,2,2-i]] # bottom row back slice
     
    slices[1][1] = [[cube[5,1,1]],[],[cube[4,1,1]]] # middle row middle slice

    remove_None(slices)
    return slices

# solve white corners
def white_corners(slices):
    # check for white corners on yellow side
    for i in range(0,3,2):
        for j in range(0,3,2):
            print(slices[2][i][j])
    


# Initialize the cube in a solved state
#cube = np.array([[[colors[i]]*3 for j in range(3)] for i in range(6)])

cube = np.array([[['w0', 'w1', 'w2'],
  ['w3', 'w4', 'w5'],
  ['w6', 'w7', 'w8']],

 [['y0', 'y1', 'y2'],
  ['y3', 'y4', 'y5'],
  ['y6', 'y7', 'y8']],

 [['r0', 'r1', 'r2'],
  ['r3', 'r4', 'r5'],
  ['r6', 'r7', 'r8']],

 [['o0', 'o1', 'o2'],
  ['o3', 'o4', 'o5'],
  ['o6', 'o7', 'o8']],

 [['b0', 'b1', 'b2'],
  ['b3', 'b4', 'b5'],
  ['b6', 'b7', 'b8']],

 [['g0', 'g1', 'g2'],
  ['g3', 'g4', 'g5'],
  ['g6', 'g7', 'g8']]])

face2pieces(cube)

slices= [[[['o6','g2','w0'],['o7','w1'],['o8','b0','w2']],
        [['g5','w3'],['w4'],['b3','w5']],
        [['r0','g8','w6'],['r1','w7'],['r2','b6','w8']]],

        [[['o3','g1'],['o4'],['o5','b1']],
        [['g4'],[],['b4']],
        [['r3','g7'],['r4'],['r5','b7']]],

        [[['o0','g0','y2'],['o1','y1'],['o2','b2','y0']],
        [['g3','y5'],['y4'],['b5','y3']],
        [['r6','g6','y8'],['r7','y7'],['r8','b8','y6']]]]

back(cube)
print(cube)


