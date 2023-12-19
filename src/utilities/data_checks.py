import json

def check_parameters(parameters):
    """
    Checks if the parameters are valid
    :return: True if the parameters are valid, False otherwise
    """
    f = open('../../data/data.json')
    
    _, activation_funcs, loss_funcs, decay_functions, regularization_func, learning_methods, weight_inits_type = json.load(f)

    for keys in parameters.keys():
        if keys == "num_layers":
            if parameters[keys] < 2:
                raise Exception("The number of layers must be greater than 1")
        elif keys == "layer_sizes":
            if len(parameters[keys]) != parameters["num_layers"]:
                raise Exception("The number of layers and the number of layer sizes must be the same")
            for layer_size in parameters[keys]:
                if layer_size < 1:
                    raise Exception("The number of units in a layer must be greater than 0")
        elif keys == "hidden_activation_funcs":
            if len(parameters[keys]) != parameters["num_layers"] - 1:
                raise Exception("The number of hidden activation functions must be the number of hidden layers")
            if parameters[keys] not in activation_funcs:
                raise Exception("The hidden activation functions are not valid")
        elif keys == "output_activation_func":
            if parameters[keys] not in activation_funcs:
                raise Exception("The output activation function is not valid")
        elif keys == "loss_func":
            if parameters[keys] not in loss_funcs:
                raise Exception("The loss function is not valid")
        elif keys == "regularization_func":
            if parameters[keys] not in regularization_func:
                raise Exception("The regularization type is not valid")
        elif keys == "learning_method":
            if parameters[keys] not in learning_methods:
                raise Exception("The learning method is not valid")
        elif keys == "weight_init_type":
            if parameters[keys] not in weight_inits_type:
                raise Exception("The weight initialization method is not valid")
        elif keys == "weight_init_range":
            if parameters[keys][0] < 0:
                raise Exception("The weight initialization range lower bound must be greater than 0")
            if parameters[keys][1] < parameters[keys][0]:
                raise Exception("The weight initialization range upper bound must be greater than the lower bound")
        elif keys == "learning_rate":
            if parameters[keys] <= 0:
                raise Exception("The learning rate must be greater than 0")
        elif keys == "learning_rate_decay_func":
            if parameters[keys] not in decay_functions:
                raise Exception("The learning rate decay function is not valid")
        elif keys == "momentum":
            if parameters[keys] < 0 or parameters[keys] > 1:
                raise Exception("The momentum must be between 0 and 1")
        elif keys == "weight_decay":
            if parameters[keys] < 0:
                raise Exception("The weight decay must be greater or equal to 0")
        elif keys == "minibatch_size":
            if parameters[keys] < 1:
                raise Exception("The minibatch size must be greater than 0")
        elif keys == "max_num_epochs":
            if parameters[keys] < 1:
                raise Exception("The max number of epochs must be greater than 0")
        else:
            print("The parameter " + keys + " is not checked")
    return True


def check_inputs_targets(inputs, targets=None, expected_inputs_dim=None, expected_targets_dim=None):
    """
    Checks if the input and target data is valid
    :param inputs: The input data
    :param targets: The target data
    :return: True if the input and target data is valid, raise exception otherwise
    """
    for input in inputs:
        if len(input) != expected_inputs_dim:
            raise Exception("The dimension of the input data is not valid")
        
    if targets is None:
        return True
    
    for target in targets:
        if len(target) != expected_targets_dim:
            raise Exception("The dimension of the target data is not valid")

    if len(inputs) != len(targets):
        raise Exception("The number of inputs and targets must be the same")
    return True
