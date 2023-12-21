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



def ms_error(predictions, targets): 
    """
    Calculate the Mean Square Error for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the Least Mean Square loss between predictions and targets.
    """
    m = len(predictions)
    squared_error = 0
    for prediction, target in zip(predictions, targets):
        squared_error += 0.5 * np.sum(np.square(np.subtract(target, prediction)))
    return squared_error / m


def ms_error_all_der(prediction, target):
    """
    Calculate the derivative of Mean Square error for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the derivative of Least Mean Square loss between predictions and targets.
    """

    return - np.subtract(target, prediction)


def mean_euclidean_error(predictions, targets): 
    """
    Calculate the Euclidean loss for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the Euclidean loss between predictions and targets.
    """
    m = len(predictions)
    euclidean_error = 0
    for prediction, target in zip(predictions, targets):
        euclidean_error += np.sqrt(np.sum(np.square(np.subtract(target, prediction))))
    return euclidean_error / m


def mean_euclidean_error_all_der(prediction, target):
    """
    Calculate the derivative of Euclidean loss for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the derivative of Euclidean loss between predictions and targets.
    """

    return - np.subtract(target, prediction) / np.linalg.norm((target-prediction), ord=2)#np.sqrt(np.sum(np.square(np.subtract(target, prediction))))


loss_funcs = {
    "mean_squared_error": Loss(ms_error, ms_error_all_der, "MS"),
    "mean_euclidean_error": Loss(mean_euclidean_error, mean_euclidean_error_all_der, "MER")
}



#def LMS_Loss(predicted, target): 
#    """
#    Calculate the Least Mean Square loss for a single training example.
#
#    :param predicted: the value predicted by the neural network.
#    :param target: the true output value.
#    :return: the result of the Least Mean Square loss between predicted and target.
#    """
#
#    return 0.5 * np.square(np.subtract(target, predicted))   
#
#
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