from utilities import datasets_utilities
from network import Network
from training import learning_methods


if __name__ == '__main__':
    # Read the datasets
    inputs, targets = datasets_utilities.read_monk("monks-1.train", rescale=True)

    # Split the datasets
    training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets = datasets_utilities.split_dataset(inputs, targets, 0.15)
    print(len(training_set_inputs))
    print(len(validation_set_inputs))

    net_parameters = {
        "input_dimension": 17,
        "num_layers": 2,
        "layers_sizes": [4, 1],
        "layers_activation_funcs": ["relu", "tanh"],
        "weight_init_type": "glorot_bengio",
        #"weight_init_range": [-0.7, 0.7]
    }

    # Create the network
    network = Network(**net_parameters)

    # Train the network
    train_parameters = {
        "network": network,
        "loss_function": "squared_error",
        "metric_function": "binary_classification_accuracy",
        "learning_rate": 0.8,
        "learning_rate_decay": False,
        "learning_rate_decay_func": "linear",
        "learning_rate_decay_epochs": 1000,
        "min_learning_rate": 0.1,
        "momentum_alpha": 0.5,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0.0
    }

    training_istance = learning_methods["sgd"](**train_parameters)
    
    training_istance.training(training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets, 1000, "all")

