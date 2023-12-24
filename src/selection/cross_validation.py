import numpy as np
from selection import grid_search as ms
from network import Network
from training import learning_methods
from utilities import stats_utilities as su


def kfold_validation(net_comb, tr_comb, dataset_inputs, dataset_targets, num_folds):

    network = Network(**net_comb)
    training_istance = learning_methods["sgd"] (network, **tr_comb)

    input_fold = np.array_split(dataset_inputs, num_folds)
    target_fold = np.array_split(dataset_targets, num_folds)

    tr_metric = np.zeros(num_folds)
    tr_loss = np.zeros(num_folds)
    vl_metric = np.zeros(num_folds)
    vl_loss = np.zeros(num_folds)

    for i in range(num_folds):
        input_train = np.concatenate(input_fold[:i] + input_fold[i+1:])
        input_val = input_fold[i]

        target_train  = np.concatenate(target_fold[:i] + target_fold[i+1:])
        target_val = target_fold[i]

        training_istance.training(input_train, target_train, input_val, target_val)
        
        tr_loss[i], tr_metric[i] = network.evaluate(input_train, target_train)
        vl_loss[i], vl_metric[i] = network.evaluate(input_val, target_val)


    stats = su.compute_stats(tr_loss, tr_metric, vl_loss, vl_metric)
 
    results = net_comb.copy()
    results.update(tr_comb)
    results.update(stats)

    return results


def double_kfolds_validation(net_hyperparams, tr_hyperparams, dataset_inputs, dataset_targets, num_folds):

    tr_loss = np.zeros(num_folds)
    tr_metric = np.zeros(num_folds)

    vl_loss = np.zeros(num_folds)
    vl_metric = np.zeros(num_folds)

    ts_loss = np.zeros(num_folds)
    ts_metric = np.zeros(num_folds)
    
    input_fold = np.array_split(dataset_inputs, num_folds)
    target_fold = np.array_split(dataset_targets, num_folds)

    for i in range(num_folds):
        input_kfold = np.concatenate(input_fold[:i] + input_fold[i+1:])
        input_test = input_fold[i]

        target_kfold  = np.concatenate(target_fold[:i] + target_fold[i+1:])
        target_test = target_fold[i]

        model_selection = ms.Grid_search(net_hyperparams, tr_hyperparams, input_kfold, target_kfold, num_folds-1)
        network, training_istance, results = model_selection.grid_search()

        training_istance.training(input_kfold, target_kfold)

        tr_loss[i] = results["tr_loss_mean"]
        tr_metric[i] = results["tr_metric_mean"]

        vl_loss[i] = results["vl_loss_mean"]
        vl_metric[i] = results["vl_metric_mean"]

        ts_loss[i], ts_metric[i] = network.evaluate(input_test, target_test)

    stats = su.compute_stats(tr_loss, tr_metric, vl_loss, vl_metric, ts_loss, ts_metric)

    return stats 