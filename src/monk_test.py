from utilities import datasets_utilities
from network import Network
import training 


if __name__ == '__main__':
    # Read the datasets
    inputs, targets = datasets_utilities.read_monk("monks-1.train", rescale=True)
    test_inputs, test_targets = datasets_utilities.read_monk("monks-1.test", rescale=True)

    # Split the datasets
    #training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets = datasets_utilities.split_dataset(inputs, targets, 0.15)

    # MONK1
    net_parameters_2 = {
        "input_dimension": 17,
        "num_layers": 2,
        "layers_sizes": [4, 1],
        "layers_activation_funcs": ["relu", "tanh"],
        "loss_func": "mean_squared_error",
        "metric_func": "binary_classification_accuracy",
        "weight_init_type": "glorot_bengio"
    }

    # Create the network
    network = Network(**net_parameters_2)

    # Train the network
    train_parameters_2 = {
        "network": network,
        "epochs": 150,
        "batch_size": "all",
        "learning_rate": 0.5,
        "momentum_alpha": 0.5,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0.0
    }

    training_istance = training.learning_methods["gd"](**train_parameters_2)
    
    #training_istance.training(inputs, targets, test_inputs, test_targets, verbose=True, plot=False)


    # MONK2
    inputs, targets = datasets_utilities.read_monk("monks-2.train", rescale=True)
    test_inputs, test_targets = datasets_utilities.read_monk("monks-2.test", rescale=True)

    net_parameters_2 = {
        "input_dimension": 17,
        "num_layers": 2,
        "layers_sizes": [3, 1],
        "layers_activation_funcs": ["relu", "tanh"],
        "loss_func": "mean_squared_error",
        "metric_func": "binary_classification_accuracy",
        "weight_init_type": "glorot_bengio"
    }

    # Create the network
    network = Network(**net_parameters_2)

    # Train the network
    train_parameters_2 = {
        "network": network,
        "epochs": 200,
        "batch_size": "all",
        "learning_rate": 0.2,
        "momentum_alpha": 0.5,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0.0
    }

    training_istance = training.learning_methods["gd"](**train_parameters_2)
    
    training_istance.training(inputs, targets, test_inputs, test_targets, verbose=True, plot=False)


    # MONK3 REG
    inputs, targets = datasets_utilities.read_monk("monks-3.train", rescale=True)
    test_inputs, test_targets = datasets_utilities.read_monk("monks-3.test", rescale=True)

    net_parameters_3 = {
        "input_dimension": 17,
        "num_layers": 2,
        "layers_sizes": [4, 1],
        "layers_activation_funcs": ["relu", "tanh"],
        "loss_func": "mean_squared_error",
        "metric_func": "binary_classification_accuracy",
        "weight_init_type": "random_uniform",
        "weight_init_range": [-0.1, 0.1]
    }

    # Create the network
    network = Network(**net_parameters_3)

    # Train the network
    train_parameters_3 = {
        "network": network,
        "epochs": 150,
        "batch_size": "all",
        "learning_rate": 0.3,
        "momentum_alpha": 0.5,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0.02
    }

    training_istance = training.learning_methods["gd"](**train_parameters_3)
    
    #training_istance.training(inputs, targets, test_inputs, test_targets, verbose=True, plot=False)