from functions import activation_functions
from utilities import network_utilities as nu
import numpy as np


class Layer:

    def __init__(self, input_dimension, num_unit, activation_func, weight_init_type, weight_init_range):
        """
        Initializes the layer
        :param input_dimension: The dimension of the input
        :param num_unit: The number of units in the layer
        :param activation_func: The activation function of the layer
        """
        self.input_dimension = input_dimension
        self.num_unit = num_unit
        self.activation = activation_functions.activation_funcs[activation_func]
        self.weight_init_type = weight_init_type
        self.weight_init_range = weight_init_range

        self.weights = nu.weights_init(self.weight_init_type, self.weight_init_range, self.num_unit, self.input_dimension)
        self.biases = nu.weights_init(self.weight_init_type, self.weight_init_range, 1, self.num_unit)[0]

        self.input = None
        self.nets = None
        self.outputs = None


    def foward_pass(self, input):
        """
        Performs a forward pass through the layer
        :param input: The input of the layer
        :return: The output of the layer
        """
        self.input = input
        #print(input)
        partial_nets = [np.dot(input, self.weights[t]) for t in range(self.num_unit)]
        #print(partial_nets)
        #print(self.biases)
        self.nets = np.add(partial_nets, self.biases)
        #print(self.nets)
        return [self.activation.function(net) for net in self.nets]
    
    
    def backward_pass(self, dErr_dOut):
        """
        Performs a backward pass through the layer
        :param delta: The delta of the next layer
        :return: The delta of the layer
        """
        dOut_dNet = [self.activation.derivative(net) for net in self.nets]
        minus_delta = np.multiply(dErr_dOut, dOut_dNet)
        gradient_w = np.zeros((self.num_unit, self.input_dimension))
        gradient_biases = minus_delta
  
        for t in range(self.num_unit):
            for u in range(self.input_dimension):
                gradient_w[t][u] = minus_delta[t] * self.input[u]

        new_dErr_dOut = np.zeros(self.input_dimension)
        for u in range(self.input_dimension):
            value = 0
            for t in range(self.num_unit):
                value += minus_delta[t] * self.weights[t][u]
            new_dErr_dOut[u] = value

        return new_dErr_dOut, gradient_biases, gradient_w
       