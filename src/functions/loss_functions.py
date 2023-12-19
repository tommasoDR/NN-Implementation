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
    def function_der(self):
        return self.__loss_der



def ms_loss(predictions, targets): 
    """
    Calculate the Least Mean Square loss for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the Least Mean Square loss between predictions and targets.
    """

    m = len(targets)
    sum_squared_diff = np.sum((predictions - targets) ** 2)
    return (1 / (2 * m)) * sum_squared_diff


def ms_loss_der(predictions, targets):
    """
    Calculate the derivative of Least Mean Square loss for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the derivative of Least Mean Square loss between predictions and targets.
    """

    m = len(targets)
    sum_squared_diff = (1 / m) * np.sum(predictions - targets)
    return sum_squared_diff



def classification_loss(predictions, targets): 
    """
    Calculate the Loss for classification.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the mean of number of different value between targets and predictions.
    """
    m = len(targets)
    diff_value = 0
    for element1, element2 in zip(predictions, targets):
        if element1 != element2:
            diff_value = diff_value + 1
    
    return diff_value/m


loss_funcs = {
    "mean_square_loss": Loss(ms_loss, ms_loss_der, "Loss"),
    "classification_loss": Loss(classification_loss, None, "Classification_Loss")
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