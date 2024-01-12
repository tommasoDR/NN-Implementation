from utilities import datasets_utilities
from network import Network
import training 
import numpy as np


if __name__ == '__main__':
    # Read the datasets
    inputs, targets = datasets_utilities.read_monk("monks-1.train", rescale=True)
    test_inputs, test_targets = datasets_utilities.read_monk("monks-1.test", rescale=True)

    # Split the datasets
    #training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets = datasets_utilities.split_dataset(inputs, targets, 0.15)

    # MONK1
    net_parameters_1 = {
        "input_dimension": 17,
        "num_layers": 2,
        "layers_sizes": [4, 1],
        "layers_activation_funcs": ["relu", "tanh"],
        "loss_func": "mean_squared_error",
        "metric_func": "binary_classification_accuracy",
        "weight_init_type": "random_uniform",
        "weight_init_range": [-0.2, 0.2]
    }

    training_losses = np.array([])
    training_accuracies = np.array([])
    test_losses = np.array([])
    test_accuracies = np.array([])

    for i in range(100):
        # Create the network
        network = Network(**net_parameters_1)

        # Train the network
        train_parameters_1 = {
            "network": network,
            "epochs": 150,
            "batch_size": "all",
            "learning_rate": 0.5,
            "momentum_alpha": 0.5,
            "nesterov_momentum": False,
            "regularization_func": "L2",
            "regularization_lambda": 0.0
        }

        training_istance = training.learning_methods["gd"](**train_parameters_1)
        """
        training_istance.training(inputs, targets, test_inputs, test_targets, verbose=True, plot=False)

        training_loss, training_metric = network.evaluate(inputs, targets)
        test_loss, test_metric = network.evaluate(test_inputs, test_targets)

        training_losses = np.append(training_losses, training_loss)
        training_accuracies = np.append(training_accuracies, training_metric)
        test_losses = np.append(test_losses, test_loss)
        test_accuracies = np.append(test_accuracies, test_metric)

    print(np.mean(training_losses), np.mean(training_accuracies), np.mean(test_losses), np.mean(test_accuracies))
    """

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
        "weight_init_type": "random_uniform",
        "weight_init_range": [-0.2, 0.2]
    }

    training_losses = np.array([])
    training_accuracies = np.array([])
    test_losses = np.array([])
    test_accuracies = np.array([])

    for i in range(100):
        # Create the network
        network = Network(**net_parameters_2)

        # Train the network
        train_parameters_2 = {
            "network": network,
            "epochs": 250,
            "batch_size": "all",
            "learning_rate": 0.2,
            "momentum_alpha": 0.5,
            "nesterov_momentum": False,
            "regularization_func": "L2",
            "regularization_lambda": 0.0
        }

        training_istance = training.learning_methods["gd"](**train_parameters_2)
        
        training_istance.training(inputs, targets, test_inputs, test_targets, verbose=True, plot=False)
        
        training_loss, training_metric = network.evaluate(inputs, targets)
        test_loss, test_metric = network.evaluate(test_inputs, test_targets)

        training_losses = np.append(training_losses, training_loss)
        training_accuracies = np.append(training_accuracies, training_metric)
        test_losses = np.append(test_losses, test_loss)
        test_accuracies = np.append(test_accuracies, test_metric)

    print(np.mean(training_losses), np.mean(training_accuracies), np.mean(test_losses), np.mean(test_accuracies))
    

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
        "weight_init_range": [-0.2, 0.2]
    }

    training_losses = np.array([])
    training_accuracies = np.array([])
    test_losses = np.array([])
    test_accuracies = np.array([])

    for i in range(100):
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
            "regularization_lambda": 0.01
        }

        training_istance = training.learning_methods["gd"](**train_parameters_3)
        """
        training_istance.training(inputs, targets, test_inputs, test_targets, verbose=True, plot=False)

        training_loss, training_metric = network.evaluate(inputs, targets)
        test_loss, test_metric = network.evaluate(test_inputs, test_targets)

        training_losses = np.append(training_losses, training_loss)
        training_accuracies = np.append(training_accuracies, training_metric)
        test_losses = np.append(test_losses, test_loss)
        test_accuracies = np.append(test_accuracies, test_metric)

    print(np.mean(training_losses), np.mean(training_accuracies), np.mean(test_losses), np.mean(test_accuracies))
    """