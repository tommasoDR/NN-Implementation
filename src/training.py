from utilities import data_checks
from functions import regularization_functions
from functions import decay_functions
from functions import loss_functions
import numpy as np
import random
import math

class SGD():

    def __init__(self, network, loss_function, learning_rate, learning_rate_decay_func, learning_rate_decay_epochs,
                 momentum_alpha, nesterov_momentum, regularization_func, regularization_lambda):
        
        parameters= {"loss_func": loss_function, "learning_method": 'sgd', "learning_rate": learning_rate, "learning_rate_decay_func": learning_rate_decay_func,
                     "learning_rate_decay_epochs": learning_rate_decay_epochs, "momentum_alpha": momentum_alpha, "nesterov_momentum": nesterov_momentum,
                     "regularization_func": regularization_func, "regularization_lambda": regularization_lambda}
        
        try:
            data_checks.check_parameters(parameters)
        except Exception as e:
            print(e); exit(1)

        self.network = network
        self.loss = loss_functions.loss_funcs[loss_function]
        self.starting_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay_functions.decay_funcs[learning_rate_decay_func]
        self.learning_rate_decay_epochs = learning_rate_decay_epochs
        self.momentum_alpha = momentum_alpha
        self.nesterov_momentum = nesterov_momentum
        self.regularization = regularization_functions.regularization_funcs[regularization_func]
        self.regularization_lambda = regularization_lambda

    
    def training(self, training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets, epochs, minibatch_size):
        """
        Trains the network
        :param training_set: The training set
        :param validation_set: The validation set
        :param epochs: The number of epochs to train for
        :param minibatch_size: The size of the minibatch
        :return: The trained network
        """
        try:
            data_checks.check_inputs_targets(training_set_inputs, training_set_targets, self.network.input_dim, self.network.output_dim)
            data_checks.check_inputs_targets(validation_set_inputs, validation_set_targets, self.network.input_dim, self.network.output_dim)
        except Exception as e:
            print(e); exit(1)
        
        training_errors = np.zeros((epochs, 1))
        validation_errors = np.zeros((epochs, 1))

        old_deltas = None
        for epoch in range(epochs):

            # shuffle training set
            training_set = list(zip(training_set_inputs, training_set_targets))
            random.shuffle(training_set)
            training_set_inputs, training_set_targets = zip(*training_set)

            # iterates on minibatches
            for minibatch_index in range(math.ceil(len(training_set_inputs) / minibatch_size)):
                mb_training_set_inputs, mb_training_set_targets = self.generate_minibatch(training_set_inputs, training_set_targets, minibatch_index, minibatch_size)
                minibatch_size = len(mb_training_set_inputs)

                outputs_mb_train = [self.network.foward_pass(input) for input in mb_training_set_inputs]

                dErr_dOut = self.loss.derivative(outputs_mb_train, mb_training_set_targets)
                gradient = self.network.backpropagation(dErr_dOut)

                deltas = self.calculate_deltas(gradient, minibatch_size)
                
                self.update_weights(deltas, old_deltas)
                
                old_deltas = deltas

            # calculate loss
            outputs_traininig = [self.network.foward_pass(input) for input in training_set_inputs]
            outputs_validation = [self.network.foward_pass(input) for input in validation_set_inputs]

            training_errors[epoch] = self.loss.function(outputs_traininig, training_set_targets)
            validation_errors[epoch] = self.loss.function(outputs_validation, validation_set_targets)

            # learning rate decay
            if self.learning_rate_decay:
                self.learning_rate = self.decay.function(self.starting_learning_rate, epoch, self.learning_rate_decay_epochs)


    def generate_minibatch(self, training_set_inputs, training_set_targets, minibatch_index, minibatch_size):
        """
        Generates a minibatch from the training set
        :param training_set_inputs: The training set inputs
        :param training_set_targets: The training set targets
        :param minibatch_index: The number of the minibatch
        :param minibatch_size: The size of the minibatch
        :return: The minibatch
        """
        start = minibatch_index * minibatch_size
        end = start + minibatch_size

        mb_training_set_inputs = [training_set_inputs[index] for index in range(start, end)]
        mb_training_set_targets = [training_set_targets[index] for index in range(start, end)]
        return mb_training_set_inputs, mb_training_set_targets


    def calculate_deltas(self, gradient, minibatch_size):
        """
        Calculates the deltas of the network from the gradient
        :param gradient: The gradient of the network
        :return: The deltas of the network
        """
        deltas = []
        for layer_index in range(self.network.num_layers):
            delta_biases = gradient[layer_index][0] / (-minibatch_size)
            delta_w = gradient[layer_index][1] / (-minibatch_size)
            deltas.append((delta_biases, delta_w))
        return deltas


    def update_weights(self, deltas, old_deltas):
        """
        Updates the weights of the network
        :param gradient: The gradient of the network
        :return: None
        """
        for layer_index in range(self.network.num_layers):
            delta_bias = deltas[layer_index][0]
            delta_w = deltas[layer_index][1]
            # regularization
            penalty_term_values = self.regularization.function(self.network.layers[layer_index].weights, self.regularization_lambda)
            self.network.layers[layer_index].weights = np.add(self.network.layers[layer_index].weights, penalty_term_values)
            # update weights
            self.network.layers[layer_index].biases = np.add(self.network.layers[layer_index].biases, self.learning_rate * delta_bias)
            self.network.layers[layer_index].weights = np.add(self.network.layers[layer_index].weights, self.learning_rate * delta_w)      
            # momentum
            old_delta_bias = old_deltas[layer_index][0]
            old_delta_w = old_deltas[layer_index][1]
            if old_deltas is not None:
                self.network.layers[layer_index].biases = np.add(self.network.layers[layer_index].biases, self.momentum_alpha * old_delta_bias)
                self.network.layers[layer_index].weights = np.add(self.network.layers[layer_index].weights, self.momentum_alpha * old_delta_w)
            
             
learning_methods = {
    'sgd': SGD
}