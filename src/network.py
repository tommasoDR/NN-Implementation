from utilities import data_checks
from functions import loss_functions
from functions import activation_functions
from layer import Layer
import numpy as np


class Network:

    def __init__(self, input_dimension, num_layers, layers_sizes, layers_activation_funcs, weight_init_type, weight_init_range=None):
        """
        Initializes the network
        :param input_dimension: The dimension of the input
        :param num_layers: The number of layers in the network (excluding the input layer)
        :param layers_sizes: The number of units in each layer (excluding the input layer)
        :param layers_activation_funcs: The activation functions for each hidden layer
        :param output_activation_func: The activation function for the output layer
        """
        self.input_dim = input_dimension
        self.output_dim = layers_sizes[-1]
        self.num_layers = num_layers
        self.layer_sizes = layers_sizes
        self.layers_activation_funcs = layers_activation_funcs
        self.weight_init_type = weight_init_type
        self.weight_init_range = weight_init_range

        try:
            self.parameters = {"num_layers": num_layers, "layers_sizes": layers_sizes, "layers_activation_funcs": layers_activation_funcs,
                                "weight_init_type": weight_init_type, "weight_init_range": weight_init_range}
            data_checks.check_param(self.parameters)
        except Exception as e:
            print(e); exit(1)

        self.layers = []
        layer_input_dim = input_dimension
        for i in range(self.num_layers):
            self.layers.append(Layer(input_dim=layer_input_dim, num_unit=layers_sizes[i], activation_func=layers_activation_funcs[i], 
                                     weight_init_type=weight_init_type, weight_init_range=weight_init_range))
            layer_input_dim = layers_sizes[i]

    
    def foward_pass(self, inputs):
        """
        Performs a forward pass through the network
        :param input: The input of the network
        :return: The output of the output layer of the network
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer.foward_pass(outputs)
        return outputs
    

    def backpropagation(self, dErr_dOut):
        """
        Performs backpropagation
        :param dErr_dOut: The derivative of the error with respect to the output of the network
        :return: The gradients of the network
        """
        for layer in reversed(self.layers):
            dErr_dOut = layer.backward_pass(dErr_dOut)


    def calculate_loss(self, inputs, targets, loss_function):
        """
        Calculates the loss of the network
        :param inputs: The inputs of the network
        :param targets: The expected outputs of the network
        :param loss_function: The loss function to use for the calculation
        :return: The loss of the network
        """
        try:
            data_checks.check_sets(inputs, self.input_dim, targets, self.output_dim)
        except Exception as e:
            print(e); exit(1)

        loss = loss_functions.loss_funcs[loss_function]
        outputs = self.forward_pass(inputs)

        return loss.function(outputs, targets)
    
    
    def calculate_outputs(self, inputs):
        """
        Calculates the outputs of the network
        :param inputs: The inputs of the network
        :return: The outputs of the network
        """
        try:
            data_checks.check_sets(inputs, self.input_dim)
        except Exception as e:
            print(e); exit(1)

        return self.foward_pass(inputs)