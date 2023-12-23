from functions import activation_functions
from utilities import network_utilities as nu
import numpy as np


class Layer:

    def __init__(self, input_dim, num_unit, activation_func, weight_init_type, weight_init_range):
        """
        Initializes the layer
        :param input_dimension: The dimension of the input
        :param num_unit: The number of units in the layer
        :param activation_func: The activation function of the layer
        """
        self.input_dimension = input_dim
        self.num_unit = num_unit
        self.activation = activation_functions.activation_funcs[activation_func]
        self.weight_init_type = weight_init_type
        self.weight_init_range = weight_init_range

        self.weights = nu.weights_init(self.num_unit, self.input_dimension, self.weight_init_type, self.weight_init_range)
        self.biases = nu.weights_init(1, self.num_unit, self.weight_init_type, self.weight_init_range)[0]

        self.inputs = None
        self.nets = None
        self.outputs = None
        self.delta = None
        self.delta_w = None
        self.delta_b = None


    def foward_pass(self, inputs):
        """
        Performs a forward pass through the layer
        :param input: The input of the layer
        :return: The output of the layer
        """
        self.inputs = inputs
        self.nets = np.dot(inputs, np.transpose(self.weights)) + self.biases
        return self.activation.function(self.nets)
    
    
    def backward_pass(self, dErr_dOut):
        """
        Performs a backward pass through the layer
        :param delta: The delta of the next layer
        :return: The delta of the layer
        """
        self.delta = - dErr_dOut * self.activation.derivative(self.nets)
        self.delta_w = np.dot(np.transpose(self.delta), self.inputs)
        self.delta_b = np.sum(self.delta, axis=0)
        return - np.dot(self.delta, self.weights)
    

    def normalize_deltas(self, learning_rate, batch_size):
        """
        Normalizes the deltas
        :param batch_size: The size of the minibatch
        """
        self.delta_w *= (learning_rate / batch_size)
        self.delta_b *= (learning_rate / batch_size)
    

    def apply_momentum(self, momentum_alpha):
        """
        Applies the momentum
        :param momentum_alpha: The momentum alpha
        """
        if self.delta_w is None or self.delta_b is None:
            return
        self.weights += momentum_alpha * self.delta_w
        self.delta_b += momentum_alpha * self.delta_b

    
    def update_weights(self, learning_rate):
        """"
        Updates the weights
        :param learning_rate: The learning rate
        """
        self.weights += learning_rate * self.delta_w
        self.biases += learning_rate * self.delta_b


    def regularize(self, regularization_lambda, regularization):
        """
        Performs regularization
        :param regularization_lambda: The regularization lambda
        """
        penalty_term = regularization.derivative(self.weights, regularization_lambda)
        self.delta_w -= penalty_term
         
       