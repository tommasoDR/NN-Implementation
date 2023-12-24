from utilities import data_checks as dc
from utilities import network_utilities as nu
from utilities import datasets_utilities as du
from utilities import stats_utilities as su  
from functions import regularization_functions
from functions import decay_functions
import numpy as np
import math


class SGD():

    def __init__(self, network, epochs, batch_size, learning_rate, learning_rate_decay, learning_rate_decay_func,
                 learning_rate_decay_epochs, min_learning_rate, momentum_alpha, nesterov_momentum, regularization_func, regularization_lambda):
        
        try:
            parameters= {"epochs": epochs, "batch_size":batch_size, "learning_rate": learning_rate, "learning_rate_decay": learning_rate_decay,
                         "learning_rate_decay_func": learning_rate_decay_func, "learning_rate_decay_epochs": learning_rate_decay_epochs,
                         "min_learning_rate": min_learning_rate, "momentum_alpha": momentum_alpha, "nesterov_momentum": nesterov_momentum,
                         "regularization_func": regularization_func,"regularization_lambda": regularization_lambda
                         }
            dc.check_param(parameters)
        except Exception as e:
            print(e); exit(1)

        self.network = network
        self.epochs = epochs
        self.batch_size = batch_size
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

    
    def training(self, tr_inputs, tr_targets, vl_set_inputs = None, vl_targets = None, verbose = False, plot = False):
        """
        Trains the network
        :param tr_set_inputs: The training set inputs
        :param tr_set_targets: The training set targets
        :param vl_set_inputs: The validation set inputs
        :param vl_set_targets: The validation set targets
        :param verbose: If True prints the results of each epoch
        :param plot: If True plots the results of the training
        :return: The trained network
        """
        # check inputs and targets dimensions of training set
        try:
            dc.check_sets(tr_inputs, self.network.input_dim, tr_targets, self.network.output_dim)
        except Exception as e:
            print(e); exit(1)

        # validation True if validation set is present
        validation = vl_set_inputs is not None and vl_targets is not None

        # check inputs and targets dimensions of validation set
        if validation:
            try:
                dc.check_sets(vl_set_inputs, self.network.input_dim, vl_targets, self.network.output_dim)
            except Exception as e:
                print(e); exit(1)

        if self.batch_size == "all":
            self.batch_size = len(tr_inputs)
        
        tr_loss = np.zeros(self.epochs)
        tr_metric = np.zeros(self.epochs)

        if validation:
            vl_loss = np.zeros(self.epochs)
            val_metric = np.zeros(self.epochs)

        for epoch in range(self.epochs):

            # one epoch of training
            tr_loss[epoch], tr_metric[epoch] = self.fitting(tr_inputs, tr_targets, self.batch_size)
                
            # apply learning rate decay   
            if self.learning_rate_decay:
                self.learning_rate = self.decay.function(self.starting_learning_rate, self.minimum_learning_rate, epoch, self.learning_rate_decay_epochs)
            
            # compute loss and metric on validation set
            if validation:
                vl_loss[epoch], val_metric[epoch] = self.validation(vl_set_inputs, vl_targets)
            
            # print epoch results
            if verbose:
                print("Epoch: " + str(epoch) + " Training metric: " + str(tr_metric[epoch]) + " Validation metric: " + str(val_metric[epoch]) +
                    "\n\tTraining loss: " + str(tr_loss[epoch]) + " Validation loss: " + str(vl_loss[epoch]))

        # plot results
        if plot:
            su.plot_results(tr_loss, vl_loss, tr_metric, val_metric, self.loss.name, self.metric.name)
    

    def fitting(self, tr_inputs, tr_targets, batch_size):
            
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
            dErr_dOut = self.network.compute_loss_derivative(b_tr_outputs, b_tr_targets)
            self.network.backpropagation(dErr_dOut)

            # restore weights after nesterov momentum
            if self.nesterov_momentum:
                nu.restore_weights(self.network, old_weights)

            # normalize deltas
            self.normalize_deltas(self.learning_rate, curr_batch_size)

            # apply regularization
            current_regularization_lambda = self.regularization_lambda * curr_batch_size / len(tr_inputs)
            self.regularize(current_regularization_lambda, self.regularization)

            # apply momentum
            self.apply_momentum(self.momentum_alpha)

            # update weights
            self.update_weights()

        # calculate loss and metric
        tr_loss, tr_metric = self.network.evaluate(tr_inputs, tr_targets)

        return tr_loss, tr_metric


    def validation(self, vl_set_inputs, vl_targets):
        """
        Computes the loss and the metric on the validation set
        :param vl_set_inputs: The validation set inputs
        :param vl_set_targets: The validation set targets
        :return: The loss and the metric on the validation set
        """
        loss, metric = self.network.evaluate(vl_set_inputs, vl_targets)

        return loss, metric


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

        b_tr_inputs = np.array([tr_inputs[i] for i in range(start, end)])
        b_tr_targets = np.array([tr_targets[i] for i in range(start, end)])
        return b_tr_inputs, b_tr_targets
    

    def normalize_deltas(self, learning_rate, batch_size):
        """
        Normalizes the deltas of the network
        :param batch_size: The size of the minibatch
        :return: None
        """
        for layer in self.network.layers:
            layer.normalize_deltas(learning_rate, batch_size)


    def apply_momentum(self, momentum_alpha):
        """
        Applies the momentum to the weights of the network
        :param old_deltas: The deltas of the previous epoch
        :return: None
        """
        for layer in self.network.layers:
            layer.apply_momentum(momentum_alpha)


    def update_weights(self, ):
        """
        Updates the weights of the network
        :param gradient: the delta of the network
        :param old_deltas: The deltas of the previous epoch
        :param current_regularization_lambda: The regularization lambda of the current epoch
        :return: None
        """
        for layer in self.network.layers:
            layer.update_weights()

    
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