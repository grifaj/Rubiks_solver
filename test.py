from cubeMini import RubiksCube
import numpy as np
from stateSolve import solve_cube
import globals
import matplotlib.pyplot as plt
import time

# intit globals
globals.init()
moves = []
for i in range(1000):
    cube = RubiksCube()
    cube.shuffle(100)
    move = len(solve_cube(cube))
    moves.append(move)

print('huh')
print(moves)
# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=moves, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Length of solution')
plt.ylabel('Frequency')
plt.title('Histogram of 2x2 solution lengths')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()