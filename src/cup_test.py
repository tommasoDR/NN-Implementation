from utilities import datasets_utilities
from network import Network
import training



if __name__ == '__main__':
    # Read the datasets
    inputs, targets, cup_inputs = datasets_utilities.read_cup()
    test_inputs, test_targets = datasets_utilities.read_cup_holdout()
    
    # Split the datasets
    #training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets = datasets_utilities.split_dataset(inputs, targets, 0.20)
    
    net_parameters = {
        "input_dimension": 10,
        "num_layers": 5,
        "layers_sizes": [150, 150, 150, 150, 3],
        "layers_activation_funcs": ["selu", "selu", "selu", "selu", "identity"],
        "loss_func": "mean_squared_error",
        "metric_func": "mean_euclidean_error",
        "weight_init_type": "glorot_bengio"
    }

    # Create the network
    network = Network(**net_parameters)

    # Train the network
    train_parameters = {
        "network": network,
        "epochs": 30000,
        "batch_size": "all",  
        "learning_rate": 0.0006,
        "learning_rate_decay": False,
        "learning_rate_decay_func": "linear",
        "learning_rate_decay_epochs": 500,
        "min_learning_rate": 0.00005,
        "momentum_alpha": 0.9,
        "nesterov_momentum": False,
        "regularization_func": "L2",
        "regularization_lambda": 0,
        "stop_if_impr_is_low": True,
        "early_stopping": False,
        "patience": 25,
        "delta_percentage": 0.03
    }

    training_istance = training.learning_methods["gd"](**train_parameters)
    
    training_istance.training(inputs, targets, verbose=True, plot=True)

    datasets_utilities.write_predictions(network.predict(cup_inputs), "predictions")


"""
# Example of cross validation
from selection import cross_validation
from selection import grid_search
import pprint

    # Read the datasets
    inputs, targets, _ = datasets_utilities.read_cup()
    
    tr_combinations = {
        "epochs": [15000],
        "batch_size": ["all"],
        "learning_rate": [0.0008, 0.0006],
        "learning_rate_decay": [False],
        "learning_rate_decay_func": ["linear"],
        "learning_rate_decay_epochs": [2000],
        "min_learning_rate": [0],
        "momentum_alpha": [0.9],
        "nesterov_momentum": [False],
        "regularization_func": ["L2"],
        "regularization_lambda": [0, 0.00001, 0.00005]
    }

    net_combinations = {
        "input_dimension": [10],
        "num_layers": [4, 5],
        "layers_sizes": [[200, 200, 200, 3], [150, 150, 150, 150, 3]],
        "layers_activation_funcs": [["relu", "relu", "relu", "identity"], ["selu", "selu", "selu", "selu", "identity"]],
        "loss_func": ["mean_squared_error"],
        "metric_func": ["mean_euclidean_error"],
        "weight_init_type": ["glorot_bengio"]
    }

    model_sel = grid_search.Grid_search(net_combinations, tr_combinations, inputs, targets, 5)
    network, training_istance, result = model_sel.grid_search(print_flag = True)

    #stats = cross_validation.double_kfolds_validation(net_combinations, tr_combinations, inputs, targets, 3)

    #pprint.pprint(stats, sort_dicts=False)

"""


"""
# Keras example
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(150, input_dim=10, activation='selu'))
model.add(Dense(150, activation='selu'))
model.add(Dense(150, activation='selu'))
model.add(Dense(150, activation='selu'))
model.add(Dense(3))

# Read the datasets
inputs, targets, _ = datasets_utilities.read_cup()

model.compile(loss='mean_squared_error', optimizer= SGD(learning_rate=0.0006, momentum=0.9), metrics=['MeanSquaredError'])

training_set_inputs, training_set_targets, validation_set_inputs, validation_set_targets = datasets_utilities.split_dataset(inputs, targets, 0.20)
history = model.fit(training_set_inputs, training_set_targets, epochs=10000, batch_size=720, verbose=1, validation_data=(validation_set_inputs, validation_set_targets))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], linestyle='dashed')
plt.ylabel('MSE')
plt.ylim(0, 8)
plt.xlabel('Epoch')
plt.legend(['Trainining', 'Validation'], loc='upper right')
plt.savefig('keras.png', bbox_inches='tight')
# model.summary()
#errors = model.evaluate(validation_set_inputs, validation_set_targets)
#print("\nErrors:")
#print(errors)
"""