import kociemba  
from cube import RubiksCube 

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


    #test solution
    return test_soln(cube, out)

def test_soln(cube, soln):
    new_soln = []
    for move in soln:
        if cube.solved(): 
            return new_soln
        cube.move2func(move)
        new_soln.append(move)
    return new_soln