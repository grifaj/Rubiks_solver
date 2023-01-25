import numpy as np
from cube import RubiksCube
import json

def heuristic(cube):
    cube_str = cube.stringify()
    if cube_str in heuristic_data:
        return heuristic_data[cube_str]
    # state not in database, calculate manhatton distance instead

    return 20 # max possible depth

def generate_next_states(state):
    cube = RubiksCube(state=state)
    next_actions = []
    for a in [(r, n, d) for r in ['h', 'v', 's'] for d in [0, 1] for n in range(3)]:
        cube = RubiksCube(state=state)
        if a[0] == 'h':
            cube.horizontal_twist(a[1], a[2])
        elif a[0] == 'v':
            cube.vertical_twist(a[1], a[2])
        elif a[0] == 's':
            cube.side_twist(a[1], a[2])
        next_actions.append((cube.stringify(),a))
    return next_actions

# iterative deepen through cube states
def ida_star(state, g, bound, path, max_depth):
    #print(g,bound,max_depth)
    cube = RubiksCube(state=state)
    if cube.solved():
        return True, g, path
    if len(path) > bound:
        return False, g, path
    min_cost = float('inf')
    best_action = None
    for next_action in generate_next_states(state):
        cube = RubiksCube(state=next_action[0])
        if cube.solved():
            path.append(next_action[1])
            return True, g+1, path
        f = g + heuristic(cube)
        #print(f)
        if f < min_cost:
            min_cost = f
            best_action = next_action
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
        
# load heuristic data
global heuristic_data
with open('heuristic.json') as f:
    heuristic_data = json.load(f)
print('data loaded')

for i in range(9,10):
    cube = RubiksCube()
    cube.shuffle(i,i)

    print(i,solve_cube(cube))

    

