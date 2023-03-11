import json

path = "C:\\Users\\Alfie\\Documents\\uni_work\\year3\\cs310\\github\\Rubiks_solver\\"
with open(path+'heuristic.json') as f:
    heuristic = json.load(f)

# find all states where two sides are the same 
heuristic = list(heuristic.keys())
count = 0
for state in heuristic:
    split = []
    for i in range(0,len(state),4):
        split.append(state[i:i+4])
    if len(set(split)) != len(split):
        count +=1

print(count)
    