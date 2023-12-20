from utilities import read_datasets
from network import Network
from training import learning_methods


if __name__ == '__main__':
    # Read the datasets
    inputs, targets = read_datasets.read_monk("monks-1.train")

    # Split the datasets
    training_set_inputs, training_set_targets = inputs[0:100], targets[0:100]
    validation_set_inputs, validation_set_targets = inputs[101:], targets[101:]

    net_parameters = {
        "input_dimension": 17,
        "num_layers": 2,
        "layer_sizes": [4, 1],
        "hidden_activation_funcs": ["relu"],
        "output_activation_func": "sigmoid",
        "weight_init_type": "random_uniform",
        "weight_init_range": [-0.7, 0.7]
    }

    # Create the network
    network = Network(**net_parameters)

    # Train the network
    train_parameters = {
        "network": network,
        "loss_function": "mean_squared_error",
        "metric_function": "binary_classification_accuracy",
        "learning_rate": 0.1,
        "learning_rate_decay": True,
        "learning_rate_decay_func": "linear",
        "learning_rate_decay_epochs": 1000,
        "minimum_learning_rate": 0.1,
        "momentum_alpha": 0.5,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0.0
    }

    training_istance = learning_methods["sgd"](**train_parameters)
    
    training_istance.training(training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets, 1000, 1)

