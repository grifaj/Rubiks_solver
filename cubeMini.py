import numpy as np
from random import choice

class RubiksCube:
   

    def __init__(
        self,
        state = None,
        array = [[[]]]
    ):
        if array != [[[]]]:
            self.array = array
            self.stringify()
        elif state is None:
            self.setSolved()
        else:
            self.setState(state)

    
    # set state of cube object from string representation
    def setState(self, state):
        blank = [[['']*2 for _ in range(2)] for _ in range(6)]
        q=0
        for i in range(6):
            for j in range(2):
                for k in range(2):
                    blank[i][j][k] = state[q]
                    q +=1
        self.array = blank

    # returns state array
    def getArray(self):
        return self.array

    # generates default array of cube along with colour choice
    def defaultArray(self):
        colours = ["w", "y", "r", "o", "b", "g"] 
        cube = [[[colours[i]]*2 for j in range(2)] for i in range(6)]
        return cube

    # put cube in default state of solved
    def setSolved(self):
        self.array = self.defaultArray()

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
        # sides have no centre so can rotate whole cube with mo problems check each side has only one colour
        for i in range(6):
            side = np.array(self.array[i]).reshape(-1)
            if len(set(side)) != 1:
                return False
        return True
    
    # print out cube nicely
    def printCube(self):
        spacing = f'{" " * (len(str(self.array[0][0])) + 2)}'
        l1 = '\n'.join(spacing + str(c) for c in np.rot90(np.array(self.array[1]),-2))
        l2 = '\n'
        for j in range(len(self.array[0])):
            l2 +=''.join(str(np.rot90(np.array(self.array[5]),-1)[j]))+ '  '
            l2 +=''.join(str(self.array[3][j]))+ '  '
            l2 +=''.join(str(np.rot90(np.array(self.array[4]))[j]))+ '  '
            l2 +=''.join(str(self.array[2][1-j][::-1]))+ '  '
            l2 +='\n'
        l3 = '\n'.join(spacing + str(c) for c in self.array[0])
        print(f'{l1}\n{l2}\n{l3}')

    def shuffle(self, numMoves):

        moves =[]
        for _ in range(numMoves):
            f = choice(['u','l','r','f','d','b'])
            d = choice(['c', 'ac'])
            moves.append((f,d))

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

        return moves

    # convert string representation of move and perform that move
    def move2func(self, move):
        (f, d) = move
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
        if f == 'x':
            self.x() if d == 'c' else self.x_prime()
        if f == 'y':
            self.y() if d == 'c' else self.y_prime()
        if f == 'z':
            self.z() if d == 'c' else self.z_prime()
        
    def front(self):
        cube = np.array(self.array)
        # make temp of top edge
        temp = np.copy(cube[3, 1, :])
        #move left edge to top edge
        cube[3, 1, :] = np.flip(cube[5, :, 1])
        # move bottom edge to left edge
        cube[5, :, 1] = cube[2, 0, :]
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
        temp = np.copy(cube[3, :, 1])
        #move front edge to top edge
        cube[3, :, 1] = cube[0, :, 1]
        # move bottom edge to front edge
        cube[0, :, 1] = cube[2, :, 1]
        # move back edge to bottom edge
        cube[2, :, 1] = np.flip(cube[1, :, 0])
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
        cube[3, :, 0] = np.flip(cube[1, :, 1])
        # move bottom edge to back edge
        cube[1, :, 1] = np.flip(cube[2, :, 0])
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
        temp = np.copy(cube[0, 1, :])
        # move left to front
        cube[0, 1, :] = cube[5, 1, :]
        # move back to left
        cube[5, 1, :] = cube[1, 1, :]
        # move right to back
        cube[1, 1, :] = cube[4, 1, :]
        # move front to right
        cube[4, 1, :] = temp
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
        cube[3, 0, :] = cube[4, :, 1]
        # move bottom to right
        cube[4, :, 1] = np.flip(cube[2, 1, :])
        # move left to bottom
        cube[2, 1, :] = cube[5, :, 0]
        # move top to left
        cube[5, :, 0] = np.flip(temp)
        # rotate back face
        cube[1, :, :] = np.rot90(cube[1, :, :],-1)
        self.array = cube


    def back_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.back()


    # axial rotatations
    def x(self):
        cube = np.array(self.array)
        # make temp of top
        temp = np.copy(cube[3, :, :])
        # move left to top
        cube[3, :, :] = np.rot90(cube[5, :, :],-1)
        # move bottom to left
        cube[5, :, :] = np.rot90(cube[2, :, :],-1)
        # move right to bottom
        cube[2, :, :] = cube[4, :, :]
        # move top to right
        cube[4, :, :] = np.rot90(temp, -1)
        # rotate front face clockwise
        cube[0, :, :] = np.rot90(cube[0, :, :],-1)
        # rotate back face anti-clockwise
        cube[1, :, :] = np.rot90(cube[1, :, :],1)
        self.array = cube

    def x_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.x()

    def y(self):
        cube = np.array(self.array)
        # make temp of front
        temp = np.copy(cube[0, :, :])
        # move right to front
        cube[0, :, :] = cube[4, :, :]
        # move back to right
        cube[4, :, :] = cube[1, :, :]
        # move left to back
        cube[1, :, :] = cube[5, :, :]
        # move front to left
        cube[5, :, :] = temp
        # rotate up face clockwise
        cube[3, :, :] = np.rot90(cube[3, :, :],-1)
        # rotate down face anti-clockwise
        cube[2, :, :] = np.rot90(cube[2, :, :],1)
        self.array = cube

    
    def y_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.y()


    def z(self):
        cube = np.array(self.array)
        # make temp of top edge
        temp = np.copy(cube[3, :, :])
        #move front edge to top edge
        cube[3, :, :] = cube[0, :, :]
        # move bottom edge to front edge
        cube[0, :, :] = cube[2, :, :]
        # move back edge to bottom edge
        cube[2, :, :] = np.flip(cube[1, :, :])
        # move top edge to back edge
        cube[1, :, :] = np.flip(temp)
        # rotate right face clockwise
        cube[4, :, :] = np.rot90(cube[4, :, :],-1)
        # rotate left face anti clockwise
        cube[5, :, :] = np.rot90(cube[5, :, :],1)
        self.array = cube

    def z_prime(self):
        # rotate clockwise three times
        for _ in range(3):
            self.z()