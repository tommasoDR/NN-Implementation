import numpy as np

class Regularizzation_function():
   
    def __init__(self, regularizer, regularizer_der, name):
        self.__regularizer = regularizer
        self.__regularizer_der = regularizer_der
        self.__name = name
    
    @property
    def name(self):
        return self.__name

    @property
    def function(self): 
        return self.__regularizer

    @property
    def derivative(self):
        return self.__regularizer_der
    

def l2_regularizer(w, lambd): 
    """
    Calculate the penalty term for L2 regularization.
    :param w: vector of weights.
    :param lambd: hyperparameter for regularization.
    :return: The penalty term for L2 regularization.
    """
    return lambd * np.sum(np.square(w), axis=1)


def l2_regularizer_der(w, lambd): 
    """""
    Calculate the derivative of penalty term for L2 regularization.
    :param w: vector of weights.
    :param lambd: hyperparameter for regularization.
    :return: The derivative of penalty term for L2 regularization.
    """ 
    return lambd * w


regularization_funcs = {
    'L2': Regularizzation_function(l2_regularizer, l2_regularizer_der, "L2_Regularization")
}