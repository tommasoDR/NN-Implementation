from utilities import data_checks as dc
from utilities import network_utilities as nu
from utilities import datasets_utilities as du
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
                 min_learning_rate, momentum_alpha, nesterov_momentum, regularization_func, regularization_lambda):
        
        try:
            parameters= {"loss_func": loss_function, "metric_function": metric_function, "learning_method": 'sgd', "learning_rate": learning_rate,
                         "learning_rate_decay": learning_rate_decay,"learning_rate_decay_func": learning_rate_decay_func, "learning_rate_decay_epochs": learning_rate_decay_epochs,
                         "min_learning_rate": min_learning_rate, "momentum_alpha": momentum_alpha, "nesterov_momentum": nesterov_momentum,
                         "regularization_func": regularization_func,"regularization_lambda": regularization_lambda
                         }
            dc.check_param(parameters)
        except Exception as e:
            print(e); exit(1)

        self.network = network
        self.loss = loss_functions.loss_funcs[loss_function]
        self.metric = metric_functions.metric_funcs[metric_function]
        self.starting_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.minimum_learning_rate = min_learning_rate
        self.decay = decay_functions.decay_funcs[learning_rate_decay_func]
        self.learning_rate_decay_epochs = learning_rate_decay_epochs
        self.momentum_alpha = momentum_alpha
        self.nesterov_momentum = nesterov_momentum
        self.regularization = regularization_functions.regularization_funcs[regularization_func]
        self.regularization_lambda = regularization_lambda

    
    def training(self, tr_inputs, tr_targets, vl_set_inputs, vl_targets, epochs, batch_size, plot = False):
        """
        Trains the network
        :param tr_set_inputs: The training set inputs
        :param tr_set_targets: The training set targets
        :param vl_set_inputs: The validation set inputs
        :param vl_set_targets: The validation set targets
        :param epochs: The number of epochs to train for
        :param minibatch_size: The size of the minibatch
        :return: The trained network
        """
        try:
            dc.check_sets(tr_inputs, self.network.input_dim, tr_targets, self.network.output_dim)
            dc.check_sets(vl_set_inputs, self.network.input_dim, vl_targets, self.network.output_dim)
        except Exception as e:
            print(e); exit(1)

        if batch_size == "all":
            batch_size = len(tr_inputs)
        
        tr_loss = np.zeros(epochs)
        vl_loss = np.zeros(epochs)
        tr_metric = np.zeros(epochs)
        val_metric = np.zeros(epochs)

        for epoch in range(epochs):

            # shuffle training set
            tr_inputs, tr_targets = du.shuffle(tr_inputs, tr_targets)

            # iterates on minibatches
            num_batch = math.ceil(len(tr_inputs) / batch_size)

            for batch_index in range(num_batch):
                b_tr_inputs, b_tr_targets = self.generate_batch(tr_inputs, tr_targets, batch_index, batch_size)
                curr_batch_size = len(b_tr_inputs)

                # nesterov momentum
                if self.nesterov_momentum:
                    old_weights = nu.get_weights(self.network)
                    self.apply_momentum(self.momentum_alpha)

                # compute deltas
                b_tr_outputs = self.network.foward_pass(b_tr_inputs)
                dErr_dOut = self.loss.derivative(b_tr_outputs, b_tr_targets)
                self.network.backpropagation(dErr_dOut)

                # restore weights after nesterov momentum
                if self.nesterov_momentum:
                    nu.restore_weights(self.network, old_weights)

                # normalize deltas
                self.normalize_deltas(curr_batch_size)

                # apply regularization
                current_regularization_lambda = self.regularization_lambda * curr_batch_size / len(tr_inputs)
                self.regularize(current_regularization_lambda, self.regularization)

                # apply momentum
                self.apply_momentum(self.momentum_alpha)
                
                # update weights
                self.update_weights(self.learning_rate)

                
            # apply learning rate decay   
            if self.learning_rate_decay:
                self.learning_rate = self.decay.function(self.starting_learning_rate, self.minimum_learning_rate, epoch, self.learning_rate_decay_epochs)
            
            # calculate loss and metric
            outputs_tr = self.network.foward_pass(tr_inputs)
            outputs_v = self.network.foward_pass(vl_set_inputs)

            tr_loss[epoch] = self.loss.function(outputs_tr, tr_targets)
            vl_loss[epoch] = self.loss.function(outputs_v, vl_targets)

            tr_metric[epoch] = self.metric.function(outputs_tr, tr_targets, self.network.layers[-1].activation.name)
            val_metric[epoch] = self.metric.function(outputs_v, vl_targets, self.network.layers[-1].activation.name)
            
            print("Epoch: " + str(epoch) + " Training metric: " + str(tr_metric[epoch]) + " Validation metric: " + str(val_metric[epoch]) +
                  "\n\tTraining loss: " + str(tr_loss[epoch]) + " Validation loss: " + str(vl_loss[epoch]))

        if plot:
            su.plot_results(tr_loss, vl_loss, tr_metric, val_metric, self.loss.name, self.metric.name)


    def generate_batch(self, tr_inputs, tr_targets, batch_index, batch_size):
        """
        Generates a minibatch from the training set
        :param training_set_inputs: The training set inputs
        :param training_set_targets: The training set targets
        :param minibatch_index: The number of the minibatch
        :param minibatch_size: The size of the minibatch
        :return: The minibatch
        """
        start = batch_index * batch_size
        end = start + batch_size

        if end > len(tr_inputs):
            end = len(tr_inputs)

        b_tr_inputs = np.array([tr_inputs[index] for index in range(start, end)])
        b_tr_targets = np.array([tr_targets[index] for index in range(start, end)])
        return b_tr_inputs, b_tr_targets
    

    def normalize_deltas(self, batch_size):
        """
        Normalizes the deltas of the network
        :param batch_size: The size of the minibatch
        :return: None
        """
        for layer in self.network.layers:
            layer.normalize_deltas(batch_size)


    def apply_momentum(self, momentum_alpha):
        """
        Applies the momentum to the weights of the network
        :param old_deltas: The deltas of the previous epoch
        :return: None
        """
        for layer in self.network.layers:
            layer.apply_momentum(momentum_alpha)


    def update_weights(self, learning_rate):
        """
        Updates the weights of the network
        :param gradient: the delta of the network
        :param old_deltas: The deltas of the previous epoch
        :param current_regularization_lambda: The regularization lambda of the current epoch
        :return: None
        """
        for layer in self.network.layers:
            layer.update_weights(learning_rate)

    
    def regularize(self, regularization_lambda, regularization):
        """
        Applies regularization to the weights of the network
        :param regularization_lambda: The regularization lambda
        :return: None
        """
        for layer in self.network.layers:
            layer.regularize(regularization_lambda, regularization)
            


learning_methods = {
    'sgd': SGD
}