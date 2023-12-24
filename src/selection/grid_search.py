from selection.cross_validation import kfold_validation
from utilities import data_checks as dc
from network import Network
from training import learning_methods
import numpy as np
import pprint
import itertools

def all_combinations(hyperparams): 
    hyperparams_keys = sorted(hyperparams.keys())  
    combo_list = list(itertools.product(*(hyperparams[k] for k in hyperparams_keys)))
    combinations = [dict(zip(hyperparams_keys, combo)) for combo in combo_list]  
    return combinations


class Grid_search():

    def __init__(self, net_hyperparams, tr_hyperparams, dataset_inputs, dataset_target, num_folds):
        
        self.net_hyperparams = net_hyperparams
        self.tr_hyperparams = tr_hyperparams 
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_target
        self.num_folds = num_folds

    def get_combinations(self):
        
        net_combs = all_combinations(self.net_hyperparams) 
        tr_combs = all_combinations(self.tr_hyperparams)

        net_combs = dc.remove_unfeasible_combinations(net_combs)
        tr_combs = dc.remove_unfeasible_combinations(tr_combs)

        return net_combs, tr_combs
    

    def grid_search(self, print_flag = False): 

        net_combs, tr_combs = self.get_combinations()

        return self.execute_search(net_combs, tr_combs, print_flag)

    @staticmethod    
    def execute_search(self, net_combs, tr_combs, print_flag):
        results = {}

        i = 1
        best_loss = np.inf
        best_net_comb = None; best_tr_comb = None
        index_best = None

        for net_comb in net_combs: 
            for tr_comb in tr_combs:
                results[f"comb_{i}"] = kfold_validation(net_comb, tr_comb, self.dataset_inputs, self.dataset_targets, self.num_folds)

                # Save the best combination
                loss_of_comb = results[f"comb_{i}"]["vl_loss_mean"]
                if loss_of_comb < best_loss:
                    best_loss = loss_of_comb
                    best_net_comb = net_comb
                    best_tr_comb = tr_comb
                    index_best = f"comb_{i}"
                
                i += 1

        network = Network(**best_net_comb)
        training_istance = learning_methods["sgd"] (network, **best_tr_comb)

        if print_flag:
            try:
                f = open("results.txt", "w")
                print(f"Best combination: {str(index_best)}\n", file=f)
                pprint.pprint(results, stream=f, sort_dicts=False)
                f.close()
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
        
        net_combs = all_combinations(self.net_hyperparams) 
        tr_combs = all_combinations(self.tr_hyperparams)

        net_combs = dc.remove_unfeasible_combinations(net_combs)
        tr_combs = dc.remove_unfeasible_combinations(tr_combs)

        net_combs = np.random.choice(net_combs, self.num_trials)
        tr_combs = np.random.choice(tr_combs, self.num_trials)

        return net_combs, tr_combs
    

    def random_grid_search(self, print_flag = False):
            
        net_combs, tr_combs = self.get_random_combinations()

        return self.execute_search(net_combs, tr_combs, print_flag)