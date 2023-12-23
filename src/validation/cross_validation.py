import numpy as np
from network import Network
from training import learning_methods


def cross_validation(net_comb, tr_comb, dataset_inputs, dataset_targets, num_folds):

    network = Network(**net_comb)
    training_istance = learning_methods["sgd"] (network, **tr_comb)

    input_fold = np.array_split(dataset_inputs, num_folds)
    target_fold = np.array_split(dataset_targets, num_folds)

    tr_metric_tot, vl_metric_tot, tr_loss_tot, vl_loss_tot = 0, 0, 0, 0

    for i in range(num_folds):
        input_train = np.concatenate(input_fold[:i] + input_fold[i+1:])
        input_val = input_fold[i]

        target_train  = np.concatenate(target_fold[:i] + target_fold[i+1:])
        target_val = target_fold[i]

        tr_metric, vl_metric, tr_loss, vl_loss = training_istance.training(input_train, target_train, input_val, target_val)
        tr_metric_tot += tr_metric
        vl_metric_tot += vl_metric
        tr_loss_tot += tr_loss
        vl_loss_tot += vl_loss

    tr_metric_mean = tr_metric / num_folds
    vl_metric_mean = vl_metric / num_folds
    tr_loss_mean = tr_loss / num_folds
    vl_loss_mean = vl_loss / num_folds
        
    results = net_comb.copy()
    results.update(tr_comb)
    results.update({'tr_metric_mean' : tr_metric_mean, 'tr_loss_mean' : tr_loss_mean, 'vl_metric_mean' : vl_metric_mean, 'vl_loss_mean' : vl_loss_mean})

    return results