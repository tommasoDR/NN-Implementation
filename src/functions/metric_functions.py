import numpy as np

class Metric(): 

    def __init__(self, func, name):
        self.__func= func
        self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def function(self):
        return  self.__func
    

def binary_classification_accuracy(predictions, targets, output_func): 
    """
    Calculate the accuracy for classification.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the mean of number of different value between targets and predictions.
    """
    if output_func == "Sigmoid":
        threshold = 0.5
        labels = [0, 1]
    elif output_func == "Tanh":
        threshold = 0
        labels = [-1, 1]
    else:
        raise Exception("Output function not supported by this metric")

    accuracy = 0
    for prediction, target in list(zip(predictions, targets)):
        if prediction[0] < threshold:
            real_prediction = labels[0]
        else:
            real_prediction = labels[1]

        if real_prediction == target[0]:
            accuracy += 1

    return accuracy / len(targets)


def ms_error(predictions, targets, _): 
    """
    Calculate the Mean Square Error for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the Least Mean Square error between predictions and targets.
    """
    return np.mean(np.sum(np.square(np.subtract(targets, predictions)), axis=1), axis=0)


def mean_euclidean_error(predictions, targets, _): 
    """
    Calculate the Euclidean error for all training examples.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the result of the Euclidean error between predictions and targets.
    """
    return np.mean(np.linalg.norm(np.subtract(targets, predictions), ord=2, axis=1), axis=0)


metric_funcs = {
    "binary_classification_accuracy": Metric(binary_classification_accuracy, "Binary Classification Accuracy"),
    "mean_squared_error": Metric(ms_error, "MSE"),
    "mean_euclidean_error": Metric(mean_euclidean_error, "MEE")
}