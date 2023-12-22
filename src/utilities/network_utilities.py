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
    for layer in reversed(network.layers):
        gradients.append((np.zeros((layer.num_unit)), np.zeros((layer.num_unit, layer.input_dimension))))
    return gradients


def get_weights(network):
    """
    Returns the weights of the network
    :param network: The network
    :return: The weights of the network
    """
    weights = []
    for layer in network.layers:
        weights_b = layer.biases
        weights_w = layer.weights
        weights.append((weights_b, weights_w))
    return weights


def restore_weights(network, weights):
    """
    Restores the weights of the network
    :param network: The network
    :param weights: The weights
    """
    for index, layer in enumerate(network.layers):
        layer.biases, layer.weights = weights[index]