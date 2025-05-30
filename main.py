from Algorithms.Random_forest.random_forest import random_forest
from Algorithms.Neural_network.Neural_network import neural_network
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

neural_network()

