import numpy as np

class Loss(): 

    def __init__(self, loss, loss_der, name):
        self.__loss = loss
        self.__loss_der = loss_der
        self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def function(self):
        return  self.__loss
    
    @property
    def derivative(self):
        return self.__loss_der


def mean_squared_error(predictions, targets): 
    """
    Calculate the Mean Square Error for all training examples.

    :param predictions: the values predicted by the neural network.
    :param targets: the true target values.
    :return: the result of the Mean Square Error between predictions and targets.
    """
    return np.mean(np.sum(np.square(np.subtract(targets, predictions)), axis=1), axis=0)


def mean_squared_error_der(predictions, targets):
    """
    Calculate the derivative of Mean Square Error for all training examples.

    :param predictions: the values predicted by the neural network.
    :param targets: the true target values.
    :return: the result of the derivative of Mean Square Error between predictions and targets.
    """
    return - np.subtract(targets, predictions) #/ len(predictions)


def mean_euclidean_error(predictions, targets): 
    """
    Calculate the Euclidean Error for all training examples.

    :param predictions: the values predicted by the neural network.
    :param targets: the true target values.
    :return: the result of the Mean Euclidean Error between predictions and targets.
    """
    return np.mean(np.linalg.norm(np.subtract(targets, predictions), ord=2, axis=1), axis=0) 


def mean_euclidean_error_der(predictions, targets):
    """
    Calculate the derivative of Euclidean Error for all training examples.

    :param predictions: the values predicted by the neural network.
    :param targets: the true target values.
    :return: the result of the derivative of Mean Euclidean Error between predictions and targets.
    """
    return - np.subtract(targets, predictions) / (mean_euclidean_error(predictions, targets)) #* len(predictions))


loss_funcs = {
    "mean_squared_error": Loss(mean_squared_error, mean_squared_error_der, "MSE"),
    "mean_euclidean_error": Loss(mean_euclidean_error, mean_euclidean_error_der, "MEE")
}


#def binaryCrossEntropy_Loss(predicted, target): 
#    """
#    Calculate the Cross Entropy loss for a single training example.
#
#    :param predicted: the value predicted by the neural network.
#    :param target: the true output value.
#    :return: the result of the Cross Entropy loss between predicted and target.
#    """
#
#    return -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))