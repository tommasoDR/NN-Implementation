from utilities import data_checks as dc
from selection import cross_validation 
from training import learning_methods
from network import Network
import numpy as np
import itertools
import pprint
import time


def all_combinations(hyperparams): 
    hyperparams_keys = sorted(hyperparams.keys())  
    combo_list = list(itertools.product(*(hyperparams[k] for k in hyperparams_keys)))
    combinations = [dict(zip(hyperparams_keys, combo)) for combo in combo_list]  
    return combinations


class Grid_search():

    def __init__(self, net_hyperparams, tr_hyperparams, dataset_inputs, dataset_target, num_folds):
        """
        Initialize the grid search.
        :param net_hyperparams: dictionary containing multiple hyperparameters for the network
        :param tr_hyperparams: dictionary containing multiple hyperparameters for the training method
        :param dataset_inputs: input dataset
        :param dataset_targets: target dataset
        :param num_folds: number of folds
        """
        self.net_hyperparams = net_hyperparams
        self.tr_hyperparams = tr_hyperparams 
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_target
        self.num_folds = num_folds

    def get_combinations(self):
        """
        Create all the possible feasible combinations of hyperparameters
        :return: two lists containing all the possible feasible combinations of hyperparameters
        """
        net_combs = all_combinations(self.net_hyperparams) 
        tr_combs = all_combinations(self.tr_hyperparams)

        net_combs = dc.remove_unfeasible_combinations(net_combs)
        tr_combs = dc.remove_unfeasible_combinations(tr_combs)

        return net_combs, tr_combs
    

    def grid_search(self, print_flag = False):
        """
        Perform a grid search on the given dataset.
        :param print_flag: if True, print the results on a file
        :return: the best network, the best training istance and the results of the grid search
        """
        net_combs, tr_combs = self.get_combinations()

        return self.execute_search(self, net_combs, tr_combs, print_flag)

    @staticmethod    
    def execute_search(self, net_combs, tr_combs, print_flag):
        i = 1
        results = {}
        best_loss = np.inf
        best_net_comb = None
        best_tr_comb = None
        index_best = None

        # Perform a k-fold cross validation for each combination of hyperparameters
        for net_comb in net_combs: 
            for tr_comb in tr_combs:
                results[f"comb_{i}"] = cross_validation.kfold_validation(net_comb, tr_comb, self.dataset_inputs, self.dataset_targets, self.num_folds)

                # If the current combination is the best one, save it
                loss_of_comb = results[f"comb_{i}"]["vl_loss_mean"]
                if loss_of_comb < best_loss:
                    best_loss = loss_of_comb
                    best_net_comb = net_comb
                    best_tr_comb = tr_comb
                    index_best = f"comb_{i}"
                
                if print_flag:
                    print(f"Combination {i} of {len(net_combs) * len(tr_combs)}")

                i += 1

        # Create the best network and the best training istance
        network = Network(**best_net_comb)
        training_istance = learning_methods["gd"] (network, **best_tr_comb)

        # Print the results on a file if print_flag is True
        if print_flag:
            try:
                t = time.localtime()
                current_time = time.strftime("%H_%M_%S", t)
                file = open(f"selection/results/results_{current_time}.txt", "w")
                print(f"Best combination: {str(index_best)}\n", file=file)
                pprint.pprint(results, stream=file, sort_dicts=False)
                file.close()
            except:
                print("Error writing results to file")

        return network, training_istance, results[index_best]

    

class Random_grid_search(Grid_search):

    def __init__(self, net_hyperparams, tr_hyperparams, dataset_inputs, dataset_target, num_trials, num_folds):
        
        self.net_hyperparams = net_hyperparams
        self.tr_hyperparams = tr_hyperparams 
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_target
        self.num_trials = num_trials
        self.num_folds = num_folds

    def get_random_combinations(self):
        """
        Create all the possible feasible combinations of hyperparameters and then randomly select a subset of them of size num_trials
        :return: two lists containing a subset of all the possible feasible combinations of hyperparameters for the grid search
        """
        net_combs = all_combinations(self.net_hyperparams) 
        tr_combs = all_combinations(self.tr_hyperparams)

        net_combs = dc.remove_unfeasible_combinations(net_combs)
        tr_combs = dc.remove_unfeasible_combinations(tr_combs)

        net_combs = np.random.choice(net_combs, self.num_trials)
        tr_combs = np.random.choice(tr_combs, self.num_trials)

        return net_combs, tr_combs
    

    def random_grid_search(self, print_flag = False):
        """
        Perform a random grid search on the given dataset.
        :param print_flag: if True, print the results on a file
        :return: the best network, the best training istance and the results of the random grid search
        """
        net_combs, tr_combs = self.get_random_combinations()

        return self.execute_search(net_combs, tr_combs, print_flag)