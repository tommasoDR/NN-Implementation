
from validation.cross_validation import cross_validation
from validation.model_combinations import all_combinations
from utilities import data_checks as dc
from network import Network
from training import learning_methods
import numpy as np
import pprint

class Model_selection():

    def __init__(self, tr_hyperparams, net_hyperparams, dataset_inputs, dataset_target, num_folds):
        
        self.tr_hyperparams = tr_hyperparams 
        self.net_hyperparams = net_hyperparams
        self.dataset_inputs = dataset_inputs
        self.dataset_targets = dataset_target
        self.num_folds = num_folds
    

    def grid_search(self): 
				
        results = {} 

        net_combs = all_combinations(self.net_hyperparams) 
        tr_combs = all_combinations(self.tr_hyperparams)

        net_combs = dc.remove_unfeasible_combinations(net_combs)
        tr_combs = dc.remove_unfeasible_combinations(tr_combs)

        i = 1
        best_loss = np.inf
        best_net_comb = None; best_tr_comb = None
        index_best = None

        for net_comb in net_combs: 
            for tr_comb in tr_combs:
                results[f"comb_{i}"] = cross_validation(net_comb, tr_comb, self.dataset_inputs, self.dataset_targets, self.num_folds)

                # Save the best combination
                loss_of_comb = results[f"comb_{i}"]["vl_loss_mean"]
                if loss_of_comb < best_loss:
                    best_loss = loss_of_comb
                    best_net_comb = net_comb
                    best_tr_comb = tr_comb
                    index_best = i
                
                i += 1
        
        network = Network(**best_net_comb)
        training_istance = learning_methods["sgd"] (network, **best_tr_comb)

        if print:
            try:
                f = open("results.txt", "w")
                print(f"Best combination: {index_best}\n", file=f)
                pprint.pprint(results, stream=f, sort_dicts=False)
                f.close()
            except:
                print("Error writing results to file")

        return network, training_istance