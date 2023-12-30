import numpy as np
import copy


def weights_init(num_unit, input_dimension, weight_init_type, weight_init_range=None):
    """
    Initializes the weights of a layer
    :param num_unit: The number of units of the layer
    :param input_dimension: The input dimension of the layer
    :param weight_init_type: The type of weight initialization
    :param weight_init_range: The range of the weights, if required by the initialization type
    :return: The initialized weights
    """
    if weight_init_type == "random_uniform":
        weights = np.random.uniform(
            low=weight_init_range[0],
            high=weight_init_range[1],
            size=(num_unit, input_dimension)
            )
    elif weight_init_type == "glorot_bengio":
        weights = np.random.uniform(
            low=-np.sqrt(6/(num_unit+input_dimension)), 
            high=np.sqrt(6/(num_unit+input_dimension)), 
            size=(num_unit, input_dimension)
            )
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
    :return: The copy of the weights of the network
    """
    weights = []
    for layer in network.layers:
        weights_b = copy.deepcopy(layer.biases)
        weights_w = copy.deepcopy(layer.weights)
        weights.append((weights_b, weights_w))
    return weights


def restore_weights(network, weights):
    """
    Restores the weights passed as parameter to the given network
    :param network: The network
    :param weights: The weights
    """
    for index, layer in enumerate(network.layers):
        layer.biases, layer.weights = weights[index]


def get_deltas(network):
    """
    Returns the deltas of the network
    :param network: The network
    :return: The copy of the deltas of the network
    """
    deltas = []
    for layer in network.layers:
        deltas.append((copy.deepcopy(layer.delta_b), copy.deepcopy(layer.delta_w)))
    return deltas