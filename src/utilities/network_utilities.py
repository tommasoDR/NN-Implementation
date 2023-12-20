import numpy as np


def weights_init(weight_init_type, weight_init_range, num_unit, input_dimension):
    """
    Initializes the weights
    :return: The initialized weights
    """
    if weight_init_type == "random_uniform":
        weights = np.random.uniform(low=weight_init_range[0], high=weight_init_range[1], size=(num_unit, input_dimension))
    return weights     


def get_empty_gradients(network):
    """
    Returns the empty gradients of the network
    :param network: The network
    :return: The empty gradients of the network
    """
    gradients = []
    for layer_index in range(network.num_layers):
        gradients.append((np.zeros((network.layers[layer_index].num_unit, 1)), np.zeros((network.layers[layer_index].num_unit, network.layers[layer_index].input_dimension))))
    return gradients