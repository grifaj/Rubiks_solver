import globals
import kociemba  
from cube import RubiksCube 
import numpy as np 
import random
import matplotlib.pyplot as plt


def convertBack(string):
    faces = [[],[],[],[],[],[]]
    order = [3,4,0,2,5,1]
    for i in range(6):
        face = string[i*9:(i+1)*9]
        out_face = []
        for j in range(3):
            row = face[j*3:(j+1)*3]
            out_face.append(list(row))
        faces[order[i]] = out_face

    cube = RubiksCube(array=faces)
    cube.printCube()

def solve_cube_kociemba(cube):
    # get colour mapping
    symbols =['F','B','D','U','R','L'] 
    colours = []
    for i in range(6):
        colours.append(cube.array[i][1][1])

    cube_string = ''
    order = [3,4,0,2,5,1]
    for i in order:
        for j in range(3):
            for k in range(3):
                cube_string += symbols[colours.index(cube.array[i][j][k])]

    #convertBack(cube_string)

    solved = kociemba.solve(cube_string)

    # convert soln back to my notation
    out = []
    solved = solved.split(' ')
    for move in solved:
        direction = 'c' if (len(move) == 1 or move[1] == '2') else 'ac'
        m = (move[0].lower(), direction)
        out.append(m)
        if move[-1] == '2':
            out.append(m)

    return out       

if __name__ == '__main__':
    solns = []
    for i in range(1000):
        cube = RubiksCube()
        rand = random.randint(0,35)
        cube.shuffle(rand)
        solns.append(len(solve_cube_kociemba(cube)))

    print('huh')
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=solns, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Length of solution')
    plt.ylabel('Frequency')
    plt.title('Bar chart of 3x3 solution lengths')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()