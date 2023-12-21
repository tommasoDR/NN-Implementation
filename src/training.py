from utilities import data_checks
from utilities import network_utilities as nu
from utilities import stats_utilities as su  
from functions import regularization_functions
from functions import decay_functions
from functions import loss_functions
from functions import metric_functions
import numpy as np
import random
import math


class SGD():

    def __init__(self, network, loss_function, metric_function, learning_rate, learning_rate_decay, learning_rate_decay_func, learning_rate_decay_epochs,
                 minimum_learning_rate, momentum_alpha, nesterov_momentum, regularization_func, regularization_lambda):
        
        parameters= {"loss_func": loss_function, "metric_function": metric_function, "learning_method": 'sgd', "learning_rate": learning_rate, "learning_rate_decay": learning_rate_decay,"learning_rate_decay_func": learning_rate_decay_func,
                     "learning_rate_decay_epochs": learning_rate_decay_epochs, "minimum_learning_rate": minimum_learning_rate, "momentum_alpha": momentum_alpha, "nesterov_momentum": nesterov_momentum,
                     "regularization_func": regularization_func, "regularization_lambda": regularization_lambda}
        
        try:
            data_checks.check_parameters(parameters)
        except Exception as e:
            print(e); exit(1)

        self.network = network
        self.loss = loss_functions.loss_funcs[loss_function]
        self.metric = metric_functions.metric_funcs[metric_function]
        self.starting_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.minimum_learning_rate = minimum_learning_rate
        self.decay = decay_functions.decay_funcs[learning_rate_decay_func]
        self.learning_rate_decay_epochs = learning_rate_decay_epochs
        self.momentum_alpha = momentum_alpha
        self.nesterov_momentum = nesterov_momentum
        self.regularization = regularization_functions.regularization_funcs[regularization_func]
        self.regularization_lambda = regularization_lambda

    
    def training(self, training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets, epochs, minibatch_size, plot = False):
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

        if minibatch_size == "all":
            minibatch_size = len(training_set_inputs)
        
        training_loss = np.zeros(epochs)
        validation_loss = np.zeros(epochs)
        training_metric = np.zeros(epochs)
        validation_metric = np.zeros(epochs)

        old_deltas = None
        for epoch in range(epochs):

            # shuffle training set
            training_set = list(zip(training_set_inputs, training_set_targets))
            random.shuffle(training_set)
            training_set_inputs, training_set_targets = zip(*training_set)

            # iterates on minibatches
            for minibatch_index in range(math.ceil(len(training_set_inputs) / minibatch_size)):
                mb_training_set_inputs, mb_training_set_targets = self.generate_minibatch(training_set_inputs, training_set_targets, minibatch_index, minibatch_size)
                current_minibatch_size = len(mb_training_set_inputs)

                # nesterov momentum
                if self.nesterov_momentum and old_deltas is not None:
                    old_weights = nu.get_weights(self.network)
                    self.apply_nesterov_momentum(old_deltas)

                gradients = nu.get_empty_gradients(self.network)
                for input, target in list(zip(mb_training_set_inputs, mb_training_set_targets)):
                    output = self.network.foward_pass(input)
                    dErr_dOut = self.loss.derivative(output, target)
                    new_gradients = self.network.backpropagation(dErr_dOut)
                    gradients = self.sum_gradients(gradients, new_gradients)

                deltas = self.calculate_deltas(gradients, current_minibatch_size)
                
                if self.nesterov_momentum and old_deltas is not None:
                    nu.restore_weights(self.network, old_weights)
                
                current_regularization_lambda = self.regularization_lambda * current_minibatch_size / len(training_set_inputs)
                
                self.update_weights(deltas, old_deltas, current_regularization_lambda)
                
                old_deltas = deltas

            # calculate loss and metric
            outputs_training = [self.network.foward_pass(input) for input in training_set_inputs]
            outputs_validation = [self.network.foward_pass(input) for input in validation_set_inputs]

            training_loss[epoch] = self.loss.function(outputs_training, training_set_targets)
            validation_loss[epoch] = self.loss.function(outputs_validation, validation_set_targets)

            training_metric[epoch] = self.metric.function(outputs_training, training_set_targets, self.network.layers[-1].activation.name)
            validation_metric[epoch] = self.metric.function(outputs_validation, validation_set_targets, self.network.layers[-1].activation.name)
            
            print("Epoch: " + str(epoch) + " Training metric: " + str(training_metric[epoch]) + " Validation metric: " + str(validation_metric[epoch]) + "\n\tTraining loss: " + str(training_loss[epoch]) + " Validation loss: " + str(validation_loss[epoch]))

            if self.learning_rate_decay:
                self.learning_rate = self.decay.function(self.starting_learning_rate, self.minimum_learning_rate, epoch, self.learning_rate_decay_epochs)

        if plot:
            su.plot_results(training_loss, validation_loss, training_metric, validation_metric, self.loss.name, self.metric.name)


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

        if end > len(training_set_inputs):
            end = len(training_set_inputs)

        mb_training_set_inputs = [training_set_inputs[index] for index in range(start, end)]
        mb_training_set_targets = [training_set_targets[index] for index in range(start, end)]
        return mb_training_set_inputs, mb_training_set_targets


    def sum_gradients(self, gradients, new_gradients):
        """
        Sums the gradients of the network
        :param gradients: The gradients of the network
        :param new_gradients: The new gradients to add to the network
        :return: The new gradients of the network
        """
        for layer_index in range(self.network.num_layers):
            gradients_biases = gradients[layer_index][0]
            gradients_w = gradients[layer_index][1]
            new_gradients_biases = new_gradients[layer_index][0]
            new_gradients_w = new_gradients[layer_index][1]
            gradients[layer_index] = (np.add(gradients_biases, new_gradients_biases), np.add(gradients_w, new_gradients_w))
        return gradients
    

    def calculate_deltas(self, gradients, current_minibatch_size):
        """
        Calculates the deltas of the network from the gradient
        :param gradient: The gradient of the network
        :return: The deltas of the network
        """
        deltas = []
        for layer_index in range(self.network.num_layers):
            gradients_biases = gradients[layer_index][0]
            gradients_w = gradients[layer_index][1]
            delta_biases = self.learning_rate * (- gradients_biases / current_minibatch_size)
            delta_w = self.learning_rate * (- gradients_w / current_minibatch_size)
            deltas.append((delta_biases, delta_w))
        return deltas


    def update_weights(self, deltas, old_deltas, current_regularization_lambda):
        """
        Updates the weights of the network
        :param gradient: the delta of the network
        :param old_deltas: The deltas of the previous epoch
        :param current_regularization_lambda: The regularization lambda of the current epoch
        :return: None
        """
        for layer_index in range(self.network.num_layers):
            delta_bias = deltas[layer_index][0]
            delta_w = deltas[layer_index][1]
            # regularization
            penalty_term_values = self.regularization.derivative(self.network.layers[layer_index].weights, current_regularization_lambda)
            self.network.layers[layer_index].weights = np.subtract(self.network.layers[layer_index].weights, penalty_term_values)
            # update weights
            self.network.layers[layer_index].biases = np.add(self.network.layers[layer_index].biases, delta_bias)
            self.network.layers[layer_index].weights = np.add(self.network.layers[layer_index].weights, delta_w)      
            # momentum
            if old_deltas is not None:
                old_delta_bias = old_deltas[layer_index][0]
                old_delta_w = old_deltas[layer_index][1]
                self.network.layers[layer_index].biases = np.add(self.network.layers[layer_index].biases, self.momentum_alpha * old_delta_bias)
                self.network.layers[layer_index].weights = np.add(self.network.layers[layer_index].weights, self.momentum_alpha * old_delta_w)
            

    def apply_nesterov_momentum(self, old_deltas):
        """
        Applies the nesterov momentum to the weights of the network
        :param old_deltas: The deltas of the previous epoch
        :return: None
        """
        for layer_index in range(self.network.num_layers):
            old_delta_bias = old_deltas[layer_index][0]
            old_delta_w = old_deltas[layer_index][1]
            self.network.layers[layer_index].biases = np.add(self.network.layers[layer_index].biases, self.momentum_alpha * old_delta_bias)
            self.network.layers[layer_index].weights = np.add(self.network.layers[layer_index].weights, self.momentum_alpha * old_delta_w)

learning_methods = {
    'sgd': SGD
}