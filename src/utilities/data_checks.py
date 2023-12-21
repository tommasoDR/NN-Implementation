import json

def check_parameters(parameters):
    """
    Checks if the parameters are valid
    :return: True if the parameters are valid, False otherwise
    """
    try:
        f = open('../data/data.json')
    except Exception as e:
        print(e); exit(1)
    
    json_data = json.load(f)

    activation_funcs = json_data["activation_funcs"]
    loss_funcs = json_data["loss_funcs"]
    metric_funcs = json_data["metric_funcs"]
    regularization_func = json_data["regularization_func"]
    learning_methods = json_data["learning_methods"]
    weight_inits_type = json_data["weight_inits_type"]
    decay_functions = json_data["decay_functions"]

    for key in parameters.keys():
        if key == "num_layers":
            if parameters[key] < 2:
                raise Exception("The number of layers must be greater than 1")
        elif key == "layer_sizes":
            if len(parameters[key]) != parameters["num_layers"]:
                raise Exception("The number of layers and the number of layer sizes must be the same")
            for layer_size in parameters[key]:
                if layer_size < 1:
                    raise Exception("The number of units in a layer must be greater than 0")
        elif key == "hidden_activation_funcs":
            if len(parameters[key]) != parameters["num_layers"] - 1:
                raise Exception("The number of hidden activation functions must be the number of hidden layers")
            for activation_func in parameters[key]:
                if str(activation_func) not in activation_funcs:
                    raise Exception("The hidden activation functions are not valid")
        elif key == "output_activation_func":
            if str(parameters[key]) not in activation_funcs:
                raise Exception("The output activation function is not valid")
        elif key == "loss_func":
            if str(parameters[key]) not in loss_funcs:
                raise Exception("The loss function is not valid")
        elif key == "metric_function":
            if str(parameters[key]) not in metric_funcs:
                raise Exception("The metric function is not valid")
        elif key == "regularization_func":
            if str(parameters[key]) not in regularization_func:
                raise Exception("The regularization type is not valid")
        elif key == "learning_method":
            if str(parameters[key]) not in learning_methods:
                raise Exception("The learning method is not valid")
        elif key == "weight_init_type":
            if str(parameters[key]) not in weight_inits_type:
                raise Exception("The weight initialization method is not valid")
            if parameters[key] == "random_uniform" and parameters["weight_init_range"] is None:
                raise Exception("The weight initialization range must be specified for random uniform initialization")
        elif key == "weight_init_range":
            if parameters[key] is None:
                continue
            if parameters[key][1] < parameters[key][0]:
                raise Exception("The weight initialization range upper bound must be greater than the lower bound")
        elif key == "learning_rate":
            if parameters[key] <= 0:
                raise Exception("The learning rate must be greater than 0")
        elif key == "learning_rate_decay":
            if parameters[key] != True and parameters[key] != False:
                raise Exception("The learning rate decay must be a boolean")
        elif key == "learning_rate_decay_func":
            if str(parameters[key]) not in decay_functions:
                raise Exception("The learning rate decay function is not valid")
        elif key == "learning_rate_decay_epochs":
            if parameters[key] < 1:
                raise Exception("The learning rate decay epochs must be greater than 0")
        elif key == "minimum_learning_rate":
            if parameters[key] <= 0:
                raise Exception("The minimum learning rate must be greater than 0")
            if parameters[key] > parameters["learning_rate"]:
                raise Exception("The minimum learning rate must be less than the learning rate")
        elif key == "momentum_alpha":
            if parameters[key] < 0:
                raise Exception("The momentum alpha must be greater or equal to 0")
        elif key == "nesterov_momentum":
            if parameters[key] != True and parameters[key] != False:
                raise Exception("The nesterov momentum must be a boolean")
        elif key == "weight_decay":
            if parameters[key] < 0:
                raise Exception("The weight decay must be greater or equal to 0")
        elif key == "minibatch_size":
            if parameters[key] < 1:
                raise Exception("The minibatch size must be greater than 0")
        elif key == "max_num_epochs":
            if parameters[key] < 1:
                raise Exception("The max number of epochs must be greater than 0")
        elif key == "regularization_lambda":
            if parameters[key] < 0:
                raise Exception("The regularization lambda must be greater or equal to 0")
        else:
            print("The parameter " + key + " is not checked")
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
