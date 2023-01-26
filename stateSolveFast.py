import numpy as np
from cube import RubiksCube
import json

# remove none placeholders from slices
def remove_None(cube):
    for a in cube:
        for b in a:
            for c in b:
                try:
                    c.remove(None)
                except ValueError:
                    pass


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

def build_heuristic_db(state, actions, max_moves = 20, heuristic = None):
    """
    Input: state - string representing the current state of the cube
           actions -list containing tuples representing the possible actions that can be taken
           max_moves - integer representing the max amount of moves alowed (default = 20) [OPTIONAL]
           heuristic - dictionary containing current heuristic map (default = None) [OPTIONAL]
    Description: create a heuristic map for determining the best path for solving a rubiks cube
    Output: dictionary containing the heuristic map
    """
    if heuristic is None:
        heuristic = {state: 0}
    que = [(state, 0)]
    node_count = sum([len(actions) ** (x + 1) for x in range(max_moves + 1)])
    with tqdm(total=node_count, desc='Heuristic DB') as pbar:
        while True:
            if not que:
                break
            s, d = que.pop()
            if d > max_moves:
                continue
            for a in actions:
                cube = RubiksCube(state=s)
                if a[0] == 'h':
                    cube.horizontal_twist(a[1], a[2])
                elif a[0] == 'v':
                    cube.vertical_twist(a[1], a[2])
                elif a[0] == 's':
                    cube.side_twist(a[1], a[2])
                a_str = cube.stringify()
                if a_str not in heuristic or heuristic[a_str] > d + 1:
                    heuristic[a_str] = d + 1
                que.append((a_str, d+1))
                pbar.update(1)
    return heuristic


# funtion to get x,y,z differece for each piece
def getSolvedPosition(piece):
    # goal state 
    solved = RubiksCube()
    solved = face2pieces(solved.array)
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if set(solved[x][y][z]) == set(piece): # check for rotation?
                    return x, y, z
    print('ruh row')
    print(piece)
    return None



def heuristic(cube):
    cube_str = cube.stringify()
    print(cube_str)
    if cube_str in heuristic_data:
        return heuristic_data[cube_str]

    print('past 6?')
    # state not in database, calculate manhatton distance instead
    pieces = face2pieces(cube.array)
    cornerDistance = 0
    edgeDistance = 0
    # Iterate through the cube
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Get the current cubelet and its goal position
                current_cubelet = pieces[i][j][k]
                #############
                if getSolvedPosition(current_cubelet) is None:
                    cube.printCube()
                #############
                goal_x, goal_y, goal_z = getSolvedPosition(current_cubelet)
                # Calculate the manhattan distance
                manDistance = abs(i - goal_x) + abs(j - goal_y) + abs(k - goal_z)

                if len(current_cubelet) == 3: # corner
                    cornerDistance += manDistance
                if len(current_cubelet) == 2: #edge
                    edgeDistance +=manDistance
                
    # Korf 1997 max of sum of corner distances and edge distances, divided by 4
    return max(cornerDistance,edgeDistance)/4
    

def generate_next_states(state):
    next_actions = []
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
def ida_star(state, g, bound, path, max_depth):
    cube = RubiksCube(state=state)
    if cube.solved():
        return True, g, path
    # stop searching if past bound
    if len(path) > bound:
        return False, g, path
    min_cost = float('inf')
    best_action = None
    for next_action in generate_next_states(state):
        cube = RubiksCube(state=next_action[0])
        if cube.solved():
            path.append(next_action[1])
            return True, g+1, path
        # calculate heuristic
        f = g + heuristic(cube)
        if f < min_cost:
            min_cost = f
            best_action = next_action
    # choose best action out of all possible
    if best_action is not None:
        if max_depth is None or min_cost < max_depth:
            max_depth = min_cost
        status, cost, new_path = ida_star(best_action[0], g+1, bound, path + [best_action[1]], max_depth)
        if status: return True, cost, new_path

    return False, cost, new_path

def solve_cube(cube):
    bound = heuristic(cube)
    path = [] 
    max_depth = None
    while True:
        status, cost, path = ida_star(cube.stringify(), 0, bound, path, max_depth)
        if status:
            return cost, path
        path = []
        bound = cost # cost is min_threshold

######################## main ##################################


# load heuristic data
global heuristic_data
with open('heuristic.json') as f:
    heuristic_data = json.load(f)
print('data loaded')

cube = RubiksCube()
heuristic(cube)


'''for i in range(1,6):
    cube = RubiksCube()
    cube.shuffle(i)

    print(i,solve_cube(cube))'''


