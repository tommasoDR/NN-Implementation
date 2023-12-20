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
    

def binary_classification_accuracy(predictions, targets): 
    """
    Calculate the Loss for classification.

    :param predicted: the values predicted by the neural network.
    :param target: the true output values.
    :return: the mean of number of different value between targets and predictions.
    """
    accuracy = 0
    for prediction, target in list(zip(predictions, targets)):
        if prediction[0] < 0.5:
            real_prediction = 0
        else:
            real_prediction = 1

        if real_prediction == target[0]:
            accuracy += 1

    return accuracy / len(targets)


metric_funcs = {
    "binary_classification_accuracy": Metric(binary_classification_accuracy, "Binary_Classification_Accuracy")
}