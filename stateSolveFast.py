import numpy as np
from cubeMini import RubiksCube
import json
from tqdm import tqdm 

def coulour_independant(state):
    num = 0
    for i in range(len(state)):
        if not state[i].isnumeric():
            state = state.replace(state[i],str(num))
            num+=1
    return state

def build_heuristic_db():

    max_moves  = 10
    cube = RubiksCube()
    state = cube.stringify()

    heuristic = {coulour_independant(state): 0}
    que = [(state, 0)]
    node_count = 3674160 # max at 14
    with tqdm(total=node_count, desc='Heuristic DB') as pbar:
        while True:
            if not que:
                break
            s, d = que.pop()
            if d > max_moves:
                continue
            for next_action in generate_next_states(s):
                a_str = next_action[0]
                # convert to colour independant representation
                a_str_num = coulour_independant(a_str)
                if a_str_num not in heuristic:
                    heuristic[a_str_num] = d + 1
                    que.append((a_str, d+1))
                    pbar.update(1)
                if heuristic[a_str_num] > d + 1:
                    heuristic[a_str_num] = d + 1

    # dump dicitonary to file
    with open('heuristic.json', 'w', encoding='utf-8') as f:
        json.dump(heuristic, f, ensure_ascii=False, indent=4)


## not in use yet ###
# funtion to get x,y,z differece for each piece
def getSolvedPosition(piece):
    # goal state 
    solved = RubiksCube()
    solved = face2pieces(solved.array)
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if set(solved[x][y][z]) == set(piece):
                    return x, y, z
    print('ruh row')
    print(piece)
    return None



def heuristic(cube):
    cube_str = coulour_independant(cube.stringify())
    if cube_str in heuristic_data:
        return heuristic_data[cube_str]

    print('should not happen',cube_str)
    return 14 ## should change

    '''# state not in database, calculate manhatton distance instead
    pieces = face2pieces(cube.array)
    cornerDistance = 0
    edgeDistance = 0
    # Iterate through the cube
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Get the current cubelet and its goal position
                current_cubelet = pieces[i][j][k]
                goal_x, goal_y, goal_z = getSolvedPosition(current_cubelet)
                # Calculate the manhattan distance
                manDistance = abs(i - goal_x) + abs(j - goal_y) + abs(k - goal_z)

                if len(current_cubelet) == 3: # corner
                    cornerDistance += manDistance
                if len(current_cubelet) == 2: #edge
                    edgeDistance +=manDistance
                
    # Korf 1997 max of sum of corner distances and edge distances, divided by 4
    return max(cornerDistance,edgeDistance)/4'''
    

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
    #print('bound',bound)
    path = [] 
    max_depth = None
    while True:
        status, cost, path = ida_star(cube.stringify(), 0, bound, path, max_depth)
        if status:
            return cost, path
        path = []
        bound = cost # cost is min_threshold

######################## main ##################################


#build_heuristic_db()


# load heuristic data
global heuristic_data
with open('heuristic.json') as f:
    heuristic_data = json.load(f)
print('data loaded')

#cube = RubiksCube()
#print(coulour_independant(cube.stringify()))

for i in range(1,10):
    cube = RubiksCube()
    moves = cube.shuffle(i)
    print(len(moves), moves)
    soln =solve_cube(cube)
    print(soln)
    if len(moves) < soln[0] or soln[0] > 14:
        print('sub-optimal')
    if heuristic(cube) > i:
        print('bad bound:', heuristic(cube))
    print()


