import tensorflow as tf
from tensorflow.keras.models import load_model
from cubeMini import RubiksCube
import stateSolve
import globals
'''
print(tf.__version__)
print(tf.config.list_physical_devices())
'''


# Load the saved model
model = load_model('cnn_model.h5')
globals.init()
stateSolve.solve_cube(RubiksCube())


for i in range(10):
    cube = RubiksCube()
    cube.shuffle(10)
    ground = stateSolve.getHeuristic(cube)
    cube_str = [int(c) for c in stateSolve.colour_independant(cube.stringify())]
    pred = model.predict(cube_str)
    print(ground, pred, ground==pred)




# Use the model for prediction
predictions = model.predict(X_test)

