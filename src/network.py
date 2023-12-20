from utilities import data_checks
from functions import loss_functions
from functions import activation_functions
from layer import Layer
import numpy as np


class Network:

    def __init__(self, input_dimension, num_layers, layer_sizes, hidden_activation_funcs, output_activation_func, weight_init_type, weight_init_range):
        """
        Initializes the network
        :param input_dimension: The dimension of the input
        :param num_layers: The number of layers in the network (excluding the input layer)
        :param layer_sizes: The number of units in each layer (excluding the input layer)
        :param hidden_activation_funcs: The activation functions for each hidden layer
        :param output_activation_func: The activation function for the output layer
        """
        self.input_dim = input_dimension
        self.output_dim = layer_sizes[-1]
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.layer_activation_funcs = hidden_activation_funcs + [output_activation_func]
        self.weight_init_type = weight_init_type
        self.weight_init_range = weight_init_range

        self.parameters = {"num_layers": num_layers, "layer_sizes": layer_sizes, "hidden_activation_funcs": hidden_activation_funcs,
                           "output_activation_func": output_activation_func, "weight_init_type": weight_init_type, "weight_init_range": weight_init_range}

        try:
            data_checks.check_parameters(self.parameters)
        except Exception as e:
            print(e); exit(1)

        self.layers = []
        layer_input_dimension = input_dimension
        for i in range(num_layers):
            self.layers.append(Layer(input_dimension=layer_input_dimension, num_unit=layer_sizes[i], activation_func=self.layer_activation_funcs[i], weight_init_type=weight_init_type, weight_init_range=weight_init_range))
            layer_input_dimension = layer_sizes[i]

    
    def foward_pass(self, input):
        """
        Performs a forward pass through the network
        :param input: The input of the network
        :return: The output of the output layer of the network
        """
        output = input
        for layer in self.layers:
            output = layer.foward_pass(output)
        return output
    

    def backpropagation(self, dErr_dOut):
        """
        Performs backpropagation
        :param dErr_dOut: The derivative of the error with respect to the output of the network
        :return: The gradients of the network
        """
        gradients = [0]*len(self.layers)
        for layer_index in reversed(range(len(self.layers))):
            dErr_dOut, gradients_biases, gradients_w = self.layers[layer_index].backward_pass(dErr_dOut)
            gradients[layer_index] = (gradients_biases, gradients_w)
        return gradients


    def calculate_loss(self, inputs, targets, loss_function):
        """
        Calculates the loss of the network
        :param inputs: The inputs of the network
        :param targets: The expected outputs of the network
        :param loss_function: The loss function to use for the calculation
        :return: The loss of the network
        """
        try:
            data_checks.check_inputs_targets(inputs, targets, self.input_dim, self.output_dim)
        except Exception as e:
            print(e); exit(1)

        loss = loss_functions.loss_funcs[loss_function]
        
        outputs = self.foward_pass(inputs)

        loss_value = 0
        for output, target in list(zip(outputs, targets)):
            loss_value += loss.function(output, target)
        return loss_value
    
    
    def calculate_outputs(self, inputs):
        """
        Calculates the outputs of the network
        :param inputs: The inputs of the network
        :return: The outputs of the network
        """
        try:
            data_checks.check_inputs_targets(inputs)
        except Exception as e:
            print(e); exit(1)

        outputs = []
        for input in inputs:
            outputs.append(self.foward_pass(input))
        return outputs

        
