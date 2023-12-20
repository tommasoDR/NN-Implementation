from utilities import read_datasets
from network import Network
from training import learning_methods


if __name__ == '__main__':
    # Read the datasets
    inputs, targets = read_datasets.read_monk("monks-1.train")

    # Split the datasets
    training_set_inputs, training_set_targets = inputs[0:80], targets[0:80]
    validation_set_inputs, validation_set_targets = inputs[81:120], targets[81:120]

    net_parameters = {
        "input_dimension": 17,
        "num_layers": 2,
        "layer_sizes": [5, 1],
        "hidden_activation_funcs": ["relu"],
        "output_activation_func": "sigmoid",
        "weight_init_type": "random_uniform",
        "weight_init_range": [0.1, 0.5]
    }

    # Create the network
    network = Network(**net_parameters)

    # Train the network
    train_parameters = {
        "network": network,
        "loss_function": "mean_squared_error",
        "metric_function": "binary_classification_accuracy",
        "learning_rate": 0.8,
        "learning_rate_decay": True,
        "learning_rate_decay_func": "linear",
        "learning_rate_decay_epochs": 400,
        "minimum_learning_rate": 0.3,
        "momentum_alpha": 0.8,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0.01
    }

    training_istance = learning_methods["sgd"](**train_parameters)
    
    training_istance.training(training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets, 400, 10)

