import numpy as np
from random import randint, choice

class RubiksCube:
   

    def __init__(
        self,
        state = None,
        array = [[[]]]
    ):

        if state is None:
                self.setSolved()
        else:
            self.setState(state)

    
    # set state of cube object from string representation
    def setState(self, state):
        blank = [[['']*3 for _ in range(3)] for _ in range(6)]
        q=0
        for i in range(6):
            for j in range(3):
                for k in range(3):
                    blank[i][j][k] = state[q]
                    q +=1
        self.array = blank

    # generates default array of cube along with colour choice
    def getArray(self):
        colours = ["w", "y", "r", "o", "b", "g"] 
        cube = [[[colours[i]]*3 for j in range(3)] for i in range(6)]
        return cube

    # put cube in default state of solved
    def setSolved(self):
        self.array = self.getArray()

    # turn cube into string represention for easy storage
    def stringify(self):
        output = ''
        for side in self.array:
            for row in side:
                for cubie in row:
                    output += cubie

        return output

    # test if cube is solved, returns boolean
    def solved(self):
        return np.array_equal(self.array, self.getArray())
    
    # print out cube nicely
    def printCube(self):
        order = [5,3,4,2]
        spacing = f'{" " * (len(str(self.array[0][0])) + 2)}'
        l1 = '\n'.join(spacing + str(c) for c in self.array[1])
        l2 = '\n'.join('  '.join(str(self.array[i][j]) for i in order) for j in range(len(self.array[0])))
        l3 = '\n'.join(spacing + str(c) for c in self.array[0])
        print(f'{l1}\n\n{l2}\n\n{l3}')

    def shuffle(self, numMoves):

        for _ in range(numMoves):
            f = choice(['u','l','r','f','d','b'])
            d = choice(['c', 'ac'])

            if f == 'u':
                self.up() if d == 'c' else self.up_prime()
            if f == 'l':
                self.left() if d == 'c' else self.left_prime()
            if f == 'r':
                self.right() if d == 'c' else self.right_prime()
            if f == 'f':
                self.front() if d == 'c' else self.front_prime()
            if f == 'd':
                self.down() if d == 'c' else self.down_prime()
            if f == 'b':
                self.back() if d == 'c' else self.back_prime()

    def front(self):
        cube = np.array(self.array)
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
        self.array = cube

    def front_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.front()

    def right(self):
        cube = np.array(self.array)
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
        self.array = cube

    def right_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.right()

    def left(self):
        cube = np.array(self.array)
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
        self.array = cube


    def left_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.left()

    def up(self):
        cube = np.array(self.array)
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
        self.array = cube

    def up_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.up()

    def down(self):
        cube = np.array(self.array)
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
        self.array = cube


    def down_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.down()

    def back(self):
        cube = np.array(self.array)
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
        self.array = cube


    def back_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.back()


    # TODO minor changes needed 
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