# Rubiks_solver

This program provides a visual solution for the 2x2 Rubik's cube, best results are obtained with an official Rubik's mini cube in natural lighting.

Requirments can be installed by running:

```
pip install -r requirements.txt
```

The program is run from `main.py` which handles the camera feed and all other files. If the camera feed doesn't appear try altering the video capture number on line 56.

`faceDetecter.py` handles detecting the state of each face and compositing it into the full state of the Rubik's cube.

`stateSolve.py` generates the solution for the state as well as the heuristic database, stored in `heuristic.hdf5`.

`showMoves.py` applies the solution to the camera and tracks for any wrong moves.

`knn-classifier.py` generates the colour classifier from `colour_data.txt` and stores the model in `knn.joblib`.

`cubie.py` creates a class for tiles that holds their colour and centre.

`cubeMini.py` creates a class for the Rubik's Cube, allowing moves and other functions to be applied to the state.

`globals.py` contains data needed between files and allows it to be transfered.