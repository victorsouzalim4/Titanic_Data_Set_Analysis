from Algorithms.Random_forest.random_forest import random_forest
from Neural_Networks_Sub.Back_Propagation.back_propagation import backPropagation
import numpy as np

#random_forest()

inputs = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])

expectedOutputs = [-1, -1, -1, -1, -1, -1, -1, 1]

nn = backPropagation(2, 3, inputs, expectedOutputs, 100000, 0.0001, "Tanh-AND-3bits", "tanh", "online")

