import numpy as np

class Decay_function(): 
    
    def __init__(self, func, name):
        self.__func = func
        self.__name = name
        
    @property
    def name(self):
        return self.__name

    @property
    def function(self):
        return self.__func
    

def linear(learning_rate, min_learning_rate, epoch, learning_rate_decay_epochs):
    """
    Computes the new learning rate
    :param learning_rate: The learning rate
    :param epoch: The current epoch
    :param learning_rate_decay_epochs: Epochs of learning rate decay before reaching the minimum learning rate
    :return: The new learning rate 
    """
    if epoch < learning_rate_decay_epochs:
        rate = epoch / learning_rate_decay_epochs
        return (1 - rate) * learning_rate + rate * min_learning_rate
    return min_learning_rate


decay_funcs = {
    'linear': Decay_function(linear, "Linear"),

}