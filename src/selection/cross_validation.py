from utilities import stats_utilities as su
from selection import grid_search as ms
from network import Network
import numpy as np
import training


def kfold_validation(net_comb, tr_comb, dataset_inputs, dataset_targets, num_folds, verbose=False):
    """
    Perform a k-fold cross validation on the given dataset.
    :param net_comb: dictionary containing the hyperparameters of the network
    :param tr_comb: dictionary containing the hyperparameters of the training method
    :param dataset_inputs: input dataset
    :param dataset_targets: target dataset
    :param num_folds: number of folds
    :param verbose: if True, print the fold number
    :return: dictionary containing the results statistics of the k-fold cross validation
    """
    input_fold = np.array_split(dataset_inputs, num_folds)
    target_fold = np.array_split(dataset_targets, num_folds)

    tr_metric = np.zeros(num_folds)
    tr_loss = np.zeros(num_folds)
    vl_metric = np.zeros(num_folds)
    vl_loss = np.zeros(num_folds)

    for i in range(num_folds):
        if verbose:
            print(f"Fold {i+1} of {num_folds}")

        # Create the network and the training istance
        network = Network(**net_comb)
        training_istance = training.learning_methods["gd"] (network, **tr_comb)

        # Split the dataset in training and validation set
        input_train = np.concatenate(input_fold[:i] + input_fold[i+1:])
        input_val = input_fold[i]

        target_train  = np.concatenate(target_fold[:i] + target_fold[i+1:])
        target_val = target_fold[i]

        # Train the network on the training set and evaluate it on the validation set
        training_istance.training(input_train, target_train, input_val, target_val)
        
        tr_loss[i], tr_metric[i] = network.evaluate(input_train, target_train)
        vl_loss[i], vl_metric[i] = network.evaluate(input_val, target_val)

    # Compute the statistics of the model
    stats = su.compute_stats(tr_loss, tr_metric, vl_loss, vl_metric)
 
    # Create a dictionary containing the results
    results = net_comb.copy()
    results.update(tr_comb)
    results.update(stats)

    return results


def double_kfolds_validation(net_hyperparams, tr_hyperparams, dataset_inputs, dataset_targets, num_folds, verbose=False):
    """
    Perform a double k-fold cross validation on the given dataset.
    :param net_hyperparams: dictionary containing multiple hyperparameters of the network
    :param tr_hyperparams: dictionary containing multiple hyperparameters of the training method
    :param dataset_inputs: input dataset
    :param dataset_targets: target dataset
    :param num_folds: number of folds
    :param verbose: if True, print the fold number
    :return: dictionary containing the results statistics of the double k-fold cross validation
    """

    tr_loss = np.zeros(num_folds)
    tr_metric = np.zeros(num_folds)

    vl_loss = np.zeros(num_folds)
    vl_metric = np.zeros(num_folds)

    ts_loss = np.zeros(num_folds)
    ts_metric = np.zeros(num_folds)
    
    input_fold = np.array_split(dataset_inputs, num_folds)
    target_fold = np.array_split(dataset_targets, num_folds)

    for i in range(num_folds):
        if verbose:
            print(f"Fold {i+1} of {num_folds} (Double k-fold)")

        # Split the dataset in training+validation and test set
        input_kfold = np.concatenate(input_fold[:i] + input_fold[i+1:])
        input_test = input_fold[i]

        target_kfold  = np.concatenate(target_fold[:i] + target_fold[i+1:])
        target_test = target_fold[i]

        # Execute the grid search to find the best model for this fold
        model_selection = ms.Grid_search(net_hyperparams, tr_hyperparams, input_kfold, target_kfold, num_folds-1)
        network, training_istance, results = model_selection.grid_search(verbose)

        # Train the network on the training+validation set and evaluate it on the test set
        training_istance.training(input_kfold, target_kfold)

        tr_loss[i] = results["tr_loss_mean"]
        tr_metric[i] = results["tr_metric_mean"]

        vl_loss[i] = results["vl_loss_mean"]
        vl_metric[i] = results["vl_metric_mean"]

        ts_loss[i], ts_metric[i] = network.evaluate(input_test, target_test)

    # Compute the statistics of all the models finded in each fold
    stats = su.compute_stats(tr_loss, tr_metric, vl_loss, vl_metric, ts_loss, ts_metric)

    return stats 