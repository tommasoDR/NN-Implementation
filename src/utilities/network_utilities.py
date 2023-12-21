import numpy as np


def weights_init(num_unit, input_dimension, weight_init_type, weight_init_range):
    """
    Initializes the weights
    :return: The initialized weights
    """
    if weight_init_type == "random_uniform":
        weights = np.random.uniform(low=weight_init_range[0], high=weight_init_range[1], size=(num_unit, input_dimension))
    elif weight_init_type == "glorot_bengio":
        weights = np.random.uniform(low=-np.sqrt(6/(num_unit+input_dimension)), high=np.sqrt(6/(num_unit+input_dimension)), size=(num_unit, input_dimension))
    return weights   


def get_empty_gradients(network):
    """
    Returns the empty gradients of the network
    :param network: The network
    :return: The empty gradients of the network
    """
    gradients = []
    for layer_index in range(network.num_layers):
        gradients.append((np.zeros((network.layers[layer_index].num_unit)), np.zeros((network.layers[layer_index].num_unit, network.layers[layer_index].input_dimension))))
    return gradients


def get_weights(network):
    """
    Returns the weights of the network
    :param network: The network
    :return: The weights of the network
    """
    weights = []
    for layer_index in range(network.num_layers):
        weights_b = network.layers[layer_index].biases
        weights_w = network.layers[layer_index].weights
        weights.append((weights_b, weights_w))
    return weights


def restore_weights(network, weights):
    """
    Restores the weights of the network
    :param network: The network
    :param weights: The weights
    """
    for layer_index in range(network.num_layers):
        network.layers[layer_index].biases = weights[layer_index][0]
        network.layers[layer_index].weights = weights[layer_index][1]