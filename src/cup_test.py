from utilities import datasets_utilities
from network import Network
from training import learning_methods
from selection import grid_search
from selection import cross_validation
import pprint


if __name__ == '__main__':
    
    # Read the datasets
    inputs, targets, _ = datasets_utilities.read_cup(normalize=True)
    """
    net_combinations = {
        "input_dimension": [10],
        "num_layers": [3,4],
        "layers_sizes": [[16,16,3],[32, 32, 32, 3]],
        "layers_activation_funcs": [["leaky_relu", "leaky_relu", "identity"],["leaky_relu", "leaky_relu", "tanh" , "identity"]],
        "loss_func": ["mean_euclidean_error"],
        "metric_func": ["mean_euclidean_error"],
        "weight_init_type": ["glorot_bengio"]
    }

    tr_combinations = {
        "epochs": [1000],
        "batch_size": [50],
        "learning_rate": [0.006, 0.004],
        "learning_rate_decay": [True, False],
        "learning_rate_decay_func": ["linear"],
        "learning_rate_decay_epochs": [2000],
        "min_learning_rate": [0.002],
        "momentum_alpha": [0.5],
        "nesterov_momentum": [False],
        "regularization_func": ["L2"],
        "regularization_lambda": [0, 0.0001]
    }

    model_sel = grid_search.Grid_search(net_combinations, tr_combinations, inputs, targets, 5)
    network, training_istance, result = model_sel.grid_search(print_flag = True)

    #stats = cross_validation.double_kfolds_validation(net_combinations, tr_combinations, inputs, targets, 3)

    #pprint.pprint(stats, sort_dicts=False)
    """
    
    

    # Split the datasets
    training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets = datasets_utilities.split_dataset(inputs, targets, 0.20)
    
    net_parameters = {
        "input_dimension": 10,
        "num_layers": 5,
        "layers_sizes": [50, 100, 100, 50, 3],
        "layers_activation_funcs": ["selu", "selu",  "selu", "selu", "identity"],
        "loss_func": "mean_squared_error",
        "metric_func": "mean_euclidean_error",
        #"weight_init_type": "random_uniform",
        #"weight_init_range": [-0.7, 0.7]
        "weight_init_type": "glorot_bengio"
    }

    # Create the network
    network = Network(**net_parameters)

    # Train the network
    train_parameters = {
        "network": network,
        "epochs": 3000,
        "batch_size": "all",  
        "learning_rate": 0.001,
        "learning_rate_decay": False,
        "learning_rate_decay_func": "linear",
        "learning_rate_decay_epochs": 600,
        "min_learning_rate": 0.000001,
        "momentum_alpha": 0.9,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0.0001,
        "early_stopping": False,
        "patience": 25,
        "delta_percentage": 0.03
    }

    training_istance = learning_methods["sgd"](**train_parameters)
    
    training_istance.training(training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets, verbose=True, plot=True)
 
