from functions import loss_functions
from functions import activation_functions
from functions import regularization_functions
import numpy as np

net = np.dot(np.array([[1, 2, -3],[0, 0, 2]]), np.transpose(np.array([[1, -1, 1], [0,0,0], [0,1,0]]))) + np.array([1, 1, 1])

derivative = activation_functions.ReLU_derivative(net)


dErrdOut= np.array([[1, 2, -3],[0, 0, 2]])

delta = - dErrdOut * derivative

print(delta)

delta_2 = np.dot(delta, np.array([[1, -1, 1], [0,0,0], [0,1,0]]))
print(delta_2)
"""
delta = np.array([[3,1,2],[1,2,6]])
input = np.array([[1],[1]])

g1 = np.outer(delta[0], input[0])
g2 = np.outer(delta[1], input[1])
print(g1)
print(g2)
print(g1+g2)
print(np.sum(delta, axis=0))
"""