from utilities import datasets_utilities
from network import Network
from training import learning_methods


if __name__ == '__main__':
    # Read the datasets
    inputs, targets, _ = datasets_utilities.read_cup()

    # Split the datasets
    training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets = datasets_utilities.split_dataset(inputs, targets, 0.15)
    
    net_parameters = {
        "input_dimension": 10,
        "num_layers": 4,
        "layer_sizes": [5, 5, 5, 3],
        "hidden_activation_funcs": ["leaky_relu", "leaky_relu", "leaky_relu"],
        "output_activation_func": "identity",
        "weight_init_type": "random_uniform",
        "weight_init_range": [-0.7, 0.7]
        #"weight_init_type": "glorot_bengio"
    }

    # Create the network
    network = Network(**net_parameters)

    # Train the network
    train_parameters = {
        "network": network,
        "loss_function": "mean_euclidean_error",
        "metric_function": "mean_euclidean_error",
        "learning_rate": 0.01,
        "learning_rate_decay": True,
        "learning_rate_decay_func": "linear",
        "learning_rate_decay_epochs": 300,
        "minimum_learning_rate": 0.0001,
        "momentum_alpha": 0.6,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0.0001
    }

    training_istance = learning_methods["sgd"](**train_parameters)
    
    training_istance.training(training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets, 1000, 170, plot=True)

