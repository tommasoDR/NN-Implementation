import itertools

def all_combinations(hyperparams): 
  	
    hyperparams_keys = sorted(hyperparams.keys())  
    combo_list = list(itertools.product(*(hyperparams[k] for k in hyperparams_keys)))
    combinations = [dict(zip(hyperparams_keys, combo)) for combo in combo_list]  
    return combinations