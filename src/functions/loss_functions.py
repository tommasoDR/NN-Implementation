import numpy as np

class Loss(): 

    def __init__(self, loss, loss_der, name):
        self.loss = loss
        self.loss_der = loss_der
        self.name = name

    @property
    def name(self):
        return self.name

    @property
    def function(self):
        return  self.loss
    
    @property
    def derivative(self):
        return self.loss_der



def ms_loss(prediction, target): 
    """
    Calculate the Least Mean Square loss for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the Least Mean Square loss between predictions and targets.
    """

    return 0.5 * np.sum(np.square(np.subtract(target, prediction)))


def ms_loss_all_der(prediction, target):
    """
    Calculate the derivative of Least Mean Square loss for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the derivative of Least Mean Square loss between predictions and targets.
    """

    return - np.subtract(target, prediction)



def binary_classification_loss(prediction, target): 
    """
    Calculate the Loss for classification.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the mean of number of different value between targets and predictions.
    """

    if prediction != target:
            return 1
    return 0


loss_funcs = {
    "mean_square_loss": Loss(ms_loss, ms_loss_all_der, "Loss"),
    "classification_loss": Loss(binary_classification_loss, None, "Classification_Loss")
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