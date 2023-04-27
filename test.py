import cv2 as cv
import numpy as np
from cube import RubiksCube

 # convert cube to slices to calculate manhattan distance, (put in cube)?   
def face2pieces(cube):
    cube = np.array(cube)
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

# remove none placeholders from slices
def remove_None(cube):
    for a in cube:
        for b in a:
            for c in b:
                try:
                    c.remove(None)
                except ValueError:
                    pass

def flatten(pieces):
    temp = []
    for layer in pieces:
        for row in layer:
            for p in row:
                temp.append(p)
    return temp


def check(pieces, test):
    for p in pieces:
        there = False
        for t in test:
            if set(p) == set(t):
                there = True
                break
        if not there:
            return False
    return True

cube = RubiksCube()
test = flatten(face2pieces(cube.array))
print(test)
print()
# flatten to just pieces
cube.shuffle(100)

pieces = flatten(face2pieces(cube.array))
print(pieces)
cube.printCube()


# break up into faces
faces = np.array(cube.array)
solns = []
num = 0
for w in range(4):
    for y in range(4):
        for r in range(4):
            for o in range(4):
                for b in range(4):
                    for g in range(4):
                        # check if all pieces are there
                        num +=1
                        pieces = flatten(face2pieces(faces))
                        if check(pieces, test):
                            if solns == []:
                                solns.append(pieces)
                            elif pieces in solns:
                                #print('dup')
                                pass
                            else:
                                solns.append(pieces)
                        np.rot90(faces[-1], 1)
                    np.rot90(faces[-2], 1)
                np.rot90(faces[-3], 1)
            np.rot90(faces[-4], 1)
        np.rot90(faces[-5], 1)
    np.rot90(faces[-6], 1)


print(len(solns))
print(num)
