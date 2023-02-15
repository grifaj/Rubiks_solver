import numpy as np
from cubeMini import RubiksCube
import json
from tqdm import tqdm 
from queue import Queue
from threading import Thread, Lock
import globals
   

def colour_independant(state):
    num = 0
    for i in range(len(state)):
        if not state[i].isnumeric():
            state = state.replace(state[i],str(num))
            num+=1
    return state

def getHeuristic(cube):
    cube_str = colour_independant(cube.stringify())
    if cube_str in globals.heuristic:
        return globals.heuristic[cube_str]

    return 14

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
def ida_star( state, g, bound, path):
    cube = RubiksCube(state=state)
    f = g + getHeuristic(cube)
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

def solve_cube(cube):
    # check heuristic is loaded
    if globals.heuristic is None:
        path = '/home/grifaj/Documents/y3project/Rubiks_solver/'
        with open(path+'heuristic.json') as f:
            globals.heuristic = json.load(f)

    path = [] 
    bound = getHeuristic(cube)
    while True:
        status, cost, path = ida_star(cube.stringify(), 0, bound, path)
        if status: return path
        bound = cost
        

# database building code hoplefuly no longer needed
def build_heuristic_db():
    max_moves  = 13
    cube = RubiksCube()
    state = cube.stringify()
    heuristic = {colour_independant(state): 0}
    total = [0, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508, 930588, 1350852, 782536, 90280, 276]
    pbar = tqdm(total=sum(total[:max_moves+1]))

    # create queue
    que = Queue()
    que.put((state, 0))
    lock = Lock()
    
    # Start the worker threads
    num_threads = 6
    threads = []
    for _ in range(num_threads):
        t = Thread(target=worker, args=(que, max_moves, heuristic, pbar, lock))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print('dumping to file')
    # dump dicitonary to file
    with open('heuristic.json', 'w', encoding='utf-8') as f:
        json.dump(heuristic, f, ensure_ascii=False, indent=4)
    f.close()
    pbar.close()

def worker(que, max_moves, heuristic, pbar, lock):
    while True:
        with lock:
            if que.empty():
                return
            s, d = que.get()
        if d >= max_moves:
            continue
        for next_action in generate_next_states(s):
            a_str = next_action[0]
            # convert to colour independant representation
            a_str_num = colour_independant(a_str)
            if a_str_num not in heuristic or heuristic[a_str_num] > d + 1:
                heuristic[a_str_num] = d + 1
                que.put((a_str, d+1))
                pbar.update(1)


######################## main ##################################


'''build_heuristic_db()
print('data dumped')'''


'''# load heuristic data
global heuristic_data
with open('heuristic.json') as f:
    heuristic_data = json.load(f)
print('data loaded')'''
