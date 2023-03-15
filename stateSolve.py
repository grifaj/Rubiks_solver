from cube import RubiksCube
import numpy as np
import json

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

def manhattan_distance(cube):
    cube = face2pieces(cube.getArray())
    solved_cube = face2pieces(RubiksCube().getArray())
    edge_dist = 0
    corner_dist = 0

    for layer in range(3):
        for row in range(3):
            for piece in range(3):
                if cube[layer][row][piece] != solved_cube[layer][row][piece]:
                    solved_pos = find_piece(solved_cube, cube[layer][row][piece])
                    dist = abs(layer - solved_pos[0]) + abs(row - solved_pos[1]) + abs(piece - solved_pos[2])
                    if len(cube[layer][row][piece]) == 3:
                        corner_dist += dist
                    elif len(cube[layer][row][piece]) == 2:
                        edge_dist += dist
                    else:
                        print('should be 0:',dist)

    return int(edge_dist/4 + corner_dist/4) 

def find_piece(cube, p):
    for layer in range(3):
        for row in range(3):
            for piece in range(3):
                if set(cube[layer][row][piece]) == set(p):
                    return [layer, row, piece]

def generate_next_states(state):
    next_actions = []
    estimated_dist = []
    for f in ['u','l','r','f','d','b']:
        for d in ['c', 'ac']:
            cube = RubiksCube(state=state)
            if f == 'u':
                cube.up() if d == 'c' else cube.up_prime()
            if f == 'l':
                cube.left() if d == 'c' else cube.left_prime()
            if f == 'r':
                cube.right() if d == 'c' else cube.right_prime()
            if f == 'f':
                cube.front() if d == 'c' else cube.front_prime()
            if f == 'd':
                cube.down() if d == 'c' else cube.down_prime()
            if f == 'b':
                cube.back() if d == 'c' else cube.back_prime()
            next_actions.append((cube.stringify(),(f,d)))
    return next_actions

# iterative deepen through cube states
def ida_star( state, g, bound, path):
    cube = RubiksCube(state=state)
    f = g + manhattan_distance(cube)
    if f > bound:
        return False, f, path
    if cube.solved():
        return True, g, path
    min_cost = float('inf')
    for next_action in generate_next_states(state):
        status, cost, new_path = ida_star(next_action[0], g+1, bound, path + [next_action[1]])
        if status:
            return True, cost, new_path
        if cost < min_cost:
            min_cost = cost
    return False, min_cost, path

def checkFront(moves, cube):
    new_moves  = []
    for move in moves:     
        prev =cube.getArray()[0]
        if type(prev) != list:
            prev = prev.tolist()
        if len(set.union(*map(set,prev))) == 1 and move[0] == 'f':
            # add y rotation and corrected move
            new_moves.append(('y', 'c'))
            new_moves.append(('l',move[1]))

            cube.move2func(new_moves[-2])
            cube.move2func(new_moves[-1])
            new_moves = new_moves + solve_cube(cube)
            return new_moves
        else:
            cube.move2func(move)
            new_moves.append(move)
    
    return new_moves

def solve_cube(cube):
    path = [] 
    bound = manhattan_distance(cube)
    while True:
        status, cost, path = ida_star(cube.stringify(), 0, bound, path)
        if status: break
        bound = cost
    
    # add rotation if front move is unchanging
    #moves = checkFront(path, cube)

    return path

if __name__ == '__main__':
    '''for i in range(100):
        cube = RubiksCube()
        print('moves', cube.shuffle(i))
        soln = solve_cube(cube)
        print('soln', soln)   
        print(len(soln))
        print()'''

    cube = RubiksCube()
    moves = [('u', 'c'), ('l', 'ac'), ('d', 'ac'), ('l', 'c'), ('l', 'ac'), ('d', 'c'), ('r', 'c'), ('l', 'ac'), ('r', 'c')]
    for move in moves:
        cube.move2func(move)
    
    soln = solve_cube(cube)
    print('soln', soln)   