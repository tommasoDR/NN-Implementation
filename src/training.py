from utilities import datasets_utilities as du
from utilities import network_utilities as nu
from utilities import stats_utilities as su
from utilities import data_checks as dc
from functions import regularization_functions
from selection import early_stopping as es
from functions import decay_functions
import numpy as np


class GD:
    def __init__(
        self,
        network,
        epochs,
        batch_size,
        learning_rate,
        momentum_alpha,
        nesterov_momentum,
        regularization_func,
        regularization_lambda,
        learning_rate_decay = False,
        learning_rate_decay_func = "linear",
        learning_rate_decay_epochs = None,
        min_learning_rate = None,
        stop_if_impr_is_low = False,
        early_stopping = False,
        patience=20,
        delta_percentage=0.03,
    ):
        """
        Initializes the gradient descent method
        :param network: The network to train
        :param epochs: The number of epochs
        :param batch_size: The size of the minibatch
        :param learning_rate: The learning rate
        :param min_learning_rate: The minimum learning rate
        :param momentum_alpha: The momentum alpha
        :param nesterov_momentum: If True applies nesterov momentum
        :param regularization_func: The regularization function
        :param regularization_lambda: The regularization lambda
        :param learning_rate_decay: If True applies learning rate decay
        :param learning_rate_decay_func: The learning rate decay function
        :param learning_rate_decay_epochs: The number of epochs after which the learning rate decays
        :param early_stopping: If True applies early stopping
        :param patience: The patience of the early stopping
        :param delta_percentage: The delta percentage of the early stopping
        """

        # Check parameters
        try:
            parameters = {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "learning_rate_decay": learning_rate_decay,
                "learning_rate_decay_func": learning_rate_decay_func,
                "learning_rate_decay_epochs": learning_rate_decay_epochs,
                "min_learning_rate": min_learning_rate,
                "momentum_alpha": momentum_alpha,
                "nesterov_momentum": nesterov_momentum,
                "regularization_func": regularization_func,
                "regularization_lambda": regularization_lambda,
                "stop_if_impr_is_low": stop_if_impr_is_low,
                "early_stopping": early_stopping,
                "patience": patience,
                "delta_percentage": delta_percentage,
            }
            dc.check_param(parameters)
        except Exception as e:
            print(e)
            exit(1)

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
        self.stop_if_impr_is_low = stop_if_impr_is_low

        if early_stopping:
            self.early_stopping = es.EarlyStopping(patience, delta_percentage)
        else:
            self.early_stopping = None


    def training(
        self,
        tr_inputs,
        tr_targets,
        vl_inputs=None,
        vl_targets=None,
        verbose=False,
        plot=False,
    ):
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
        # Check inputs and targets dimensions of training set
        try:
            dc.check_sets(
                tr_inputs, self.network.input_dim, tr_targets, self.network.output_dim
            )
        except Exception as e:
            print(e)
            exit(1)

        # Validation True if validation set is present
        validation = vl_inputs is not None and vl_targets is not None

        # Check inputs and targets dimensions of validation set
        if validation:
            try:
                dc.check_sets(
                    vl_inputs,
                    self.network.input_dim,
                    vl_targets,
                    self.network.output_dim
                )
            except Exception as e:
                print(e)
                exit(1)

        if self.batch_size == "all":
            self.batch_size = len(tr_inputs)

        tr_loss = np.array([])
        tr_metric = np.array([])

        if validation:
            vl_loss = np.array([])
            vl_metric = np.array([])

        # training loop
        for epoch in range(self.epochs):

            # one epoch of training
            tr_loss_epoch, tr_metric_epoch = self.fitting(
                tr_inputs, tr_targets, self.batch_size
            )

            if self.stop_if_impr_is_low:
                if epoch > 1 and abs(tr_loss[-1] - tr_loss_epoch) < 1e-6:
                    break

            # store loss and metric of the epoch
            tr_loss = np.append(tr_loss, tr_loss_epoch)
            tr_metric = np.append(tr_metric, tr_metric_epoch)

            # apply learning rate decay
            if self.learning_rate_decay:
                self.learning_rate = self.decay.function(
                    self.starting_learning_rate,
                    self.minimum_learning_rate,
                    epoch,
                    self.learning_rate_decay_epochs,
                )

            # compute loss and metric on validation set
            if validation:
                vl_loss_epoch, vl_metric_epoch = self.validation(vl_inputs, vl_targets)
                vl_loss = np.append(vl_loss, vl_loss_epoch)
                vl_metric = np.append(vl_metric, vl_metric_epoch)
                
                # check early stopping
                if self.early_stopping and self.early_stopping.early_stop(vl_loss[epoch]):
                    if verbose:
                        print("Early stopping at epoch: " + str(epoch))
                    break

            # print epoch results
            if verbose:
                if validation:
                    print(
                        "Epoch: "
                        + str(epoch)
                        + "\tTraining metric: "
                        + "{:.4f}".format(tr_metric[epoch])
                        + " Validation metric: "
                        + "{:.4f}".format(vl_metric[epoch])
                        + "\n\t\tTraining loss: "
                        + "{:.4f}".format(tr_loss[epoch])
                        + " Validation loss: "
                        + "{:.4f}".format(vl_loss[epoch])
                        + "\n"
                    )
                else:
                    print(
                        "Epoch: "
                        + str(epoch)
                        + "\tTraining metric: "
                        + "{:.4f}".format(tr_metric[epoch])
                        + "\tTraining loss: "
                        + "{:.4f}".format(tr_loss[epoch])
                        + "\n"
                    )
                    
        # plot results
        if plot:
            if validation:
                su.plot_results(
                    tr_loss,
                    tr_metric,
                    vl_loss,
                    vl_metric,
                    self.network.loss.name,
                    self.network.metric.name,
                )
            else:
                su.plot_results(
                    tr_loss,
                    tr_metric,
                    None,
                    None,
                    self.network.loss.name,
                    self.network.metric.name,
                )


    def fitting(self, tr_inputs, tr_targets, batch_size):
        """
        Executes one epoch of training
        """

        # shuffle training set
        tr_inputs, tr_targets = du.shuffle(tr_inputs, tr_targets)

        # iterates on minibatches
        num_batch = int(np.ceil(len(tr_inputs) / batch_size))

        for batch_index in range(num_batch):
            # generate minibatch
            b_tr_inputs, b_tr_targets = self.generate_batch(tr_inputs, tr_targets, batch_index, batch_size)
            curr_batch_size = len(b_tr_inputs)

            # get deltas for momentum
            old_deltas = nu.get_deltas(self.network)

            # nesterov momentum
            if self.nesterov_momentum:
                old_weights = nu.get_weights(self.network)
                self.apply_nest_momentum(self.momentum_alpha, old_deltas)

            # compute deltas
            b_tr_outputs = self.network.foward_pass(b_tr_inputs)
            dErr_dOut = self.network.compute_loss_derivative(b_tr_outputs, b_tr_targets)
            self.network.backpropagation(dErr_dOut)

            # restore weights after nesterov momentum
            if self.nesterov_momentum:
                nu.restore_weights(self.network, old_weights)

            # normalize deltas
            self.normalize_deltas(self.learning_rate, curr_batch_size)

            # apply momentum
            if self.momentum_alpha > 0:
                self.apply_momentum(self.momentum_alpha, old_deltas)

            # apply regularization
            if self.regularization_lambda > 0:
                current_regularization_lambda = (self.regularization_lambda * curr_batch_size / len(tr_inputs))
                self.regularize(current_regularization_lambda, self.regularization)

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

        b_tr_inputs = np.array(tr_inputs[start:end])
        b_tr_targets = np.array(tr_targets[start:end])
        return b_tr_inputs, b_tr_targets


    def normalize_deltas(self, learning_rate, batch_size):
        """
        Normalizes the deltas of the network
        :param batch_size: The size of the minibatch
        :return: None
        """
        for layer in self.network.layers:
            layer.normalize_deltas(learning_rate, batch_size)


    def apply_momentum(self, momentum_alpha, old_deltas):
        """
        Applies the momentum to the weights of the network
        :param old_deltas: The deltas of the previous epoch
        :return: None
        """
        for i, layer in enumerate(self.network.layers):
            layer.apply_momentum(momentum_alpha, old_deltas[i])

    
    def apply_nest_momentum(self, momentum_alpha, old_deltas):
        """
        Applies the momentum to the weights of the network
        :param old_deltas: The deltas of the previous epoch
        :return: None
        """
        for i, layer in enumerate(self.network.layers):
            layer.apply_nest_momentum(momentum_alpha, old_deltas[i])


    def update_weights(self):
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


learning_methods = {"gd": GD}