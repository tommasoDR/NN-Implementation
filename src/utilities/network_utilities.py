import numpy as np


def weights_init(weight_init_type, weight_init_range, input_dimension, num_unit):
    """
    Initializes the weights
    :return: The initialized weights
    """
    if weight_init_type == "random":
        weights = np.random.uniform(low=weight_init_range[0], high=weight_init_range[1], size=(input_dimension, num_unit))
    return weights     
