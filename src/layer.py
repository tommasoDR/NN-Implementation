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

        self.weights = nu.weights_init(
            self.num_unit,
            self.input_dimension,
            self.weight_init_type,
            self.weight_init_range,
        )

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
        :param inputs: The inputs of the layer
        :return: The outputs of the layer
        """
        self.inputs = inputs
        self.nets = np.dot(inputs, np.transpose(self.weights)) + self.biases
        return self.activation.function(self.nets)


    def backward_pass(self, dErr_dOut):
        """
        Performs a backward pass through the layer
        :param dErr_dOut: The derivative of the error with respect to the output of the layer
        :return: The derivative of the error for the previous layer
        """
        self.delta = -dErr_dOut * self.activation.derivative(self.nets)
        self.delta_w = np.dot(np.transpose(self.delta), self.inputs)
        self.delta_b = np.sum(self.delta, axis=0)
        return -np.dot(self.delta, self.weights)


    def normalize_deltas(self, learning_rate, batch_size):
        """
        Normalizes the deltas of the layer
        :param learning_rate: The learning rate
        :param batch_size: The size of the minibatch
        """
        self.delta_w *= learning_rate / batch_size
        self.delta_b *= learning_rate / batch_size
    

    def apply_momentum(self, momentum_alpha, old_delta):
        """
        Applies the momentum
        :param momentum_alpha: The momentum alpha
        """
        old_delta_bias, old_delta_weight = old_delta
        if old_delta_bias is None or old_delta_weight is None:
            return
        self.delta_b += (momentum_alpha * old_delta_bias)
        self.delta_w += (momentum_alpha * old_delta_weight)
    

    def apply_nest_momentum(self, momentum_alpha, old_delta):
        """
        Applies the nest momentum
        :param momentum_alpha: The momentum alpha
        :param old_delta: The old delta
        """
        old_delta_bias, old_delta_weight = old_delta
        if old_delta_bias is None or old_delta_weight is None:
            return
        self.biases += (momentum_alpha * old_delta_bias)
        self.weights += (momentum_alpha * old_delta_weight)


    def update_weights(self):
        """ 
        Updates the weights applying the deltas
        """
        self.biases += self.delta_b
        self.weights += self.delta_w


    def regularize(self, regularization_lambda, regularization):
        """
        Performs regularization
        :param regularization_lambda: The regularization lambda
        :param regularization: The regularization function
        """
        penalty_term = regularization.derivative(self.weights, regularization_lambda)
        self.weights -= penalty_term