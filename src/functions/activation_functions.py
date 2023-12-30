import numpy as np

class Activation_function(): 
    
    def __init__(self, func, derivfunc, name):
        self.__func = func
        self.__name = name
        self.__derivfunc = derivfunc

    @property
    def name(self):
        return self.__name

    @property
    def function(self):
        return self.__func
    
    @property
    def derivative(self): 
        return self.__derivfunc
    


def identity(x): 
    """"
    Compute the identity function.

    :param x: a vector of values.
    :return: the result of the linear function applied to x.
    """
    return x


def identity_derivative(_): 
    """
    Compute the derivative of identity function.
    """
    return 1
    

def sigmoid(x, alpha=1):
    """
    Compute the sigmoid function.

    :param x: net values.
    :param alpha: a scalar parameter that influences the slope of the sigmoid curve.
    :return: the result of the sigmoid function applied to x.
    """
    return 1 / (1 + np.exp(-(alpha*x)))


def sigmoid_derivative(x, alpha=1):
    """"
    Compute the derivative of sigmoid function.
    :param x: net values.
    :param alpha: a scalar parameter that influences the slope of the sigmoid curve.
    :return: the result of the derivative of sigmoid function applied to x.
    """
    sig_x = sigmoid(x, alpha)
    return alpha * sig_x * (1 - sig_x)

    
def tanh(x, alpha=1): 
    """
    Compute the tanh function.
    :param x: net values.
    :param alpha: a scalar parameter that influences the slope of the tanh curve.
    :return: the result of the tanh function applied to x.
    """
    return np.tanh(alpha * x)


def tanh_derivative(x, alpha=1):
    """"
    Compute the derivative of tanh function.
    :param x: net values.
    :param alpha: a scalar parameter that influences the slope of the tanh curve.
    :return: the result of the derivative of tanh function applied to x.
    """
    sech_squared = np.square(np.divide(1, np.cosh(alpha * x)))
    return alpha * sech_squared


def ReLU(x): 
    """
    Compute the ReLU function.
    :param x: net values.
    :return: the result of the ReLU function applied to x.
    """
    return np.maximum(0,x)


def ReLU_derivative(x):
    """
    Compute the derivative of ReLU function.
    :param x: net values.
    :return: the result of the derivative of ReLU function applied to x.
    """
    return np.where(x <= 0, 0, 1)


def leaky_ReLU(x):
    """
    Compute the leaky ReLU function.
    :param x: net values.
    :return: the result of the leaky ReLU function applied to x.
    """
    return np.maximum(0.01*x, x)


def leaky_ReLU_derivative(x):
    """
    Compute the derivative of leaky ReLU function.
    :param x: net values.
    :return: the result of the derivative of leaky ReLU function applied to x.
    """
    return np.where(x <= 0, 0.01, 1) 


def SELU(x, lambd = 1.0507, alpha = 1.6732):
    """
    Compute the SELU function.
    :param x: net values.
    :param lambd: a scalar parameter that influences the slope of the SELU curve.
    :param alpha: a scalar parameter that influences the slope of the SELU curve.
    :return: the result of the SELU function applied to x.
    """
    return np.where(x > 0, lambd * x, lambd * alpha * (np.exp(x) - 1))
    

def SELU_derivative(x, lambd = 1.0507, alpha = 1.6732):
    """
    Compute the derivative of SELU function.
    :param x: net values.
    :param lambd: a scalar parameter that influences the slope of the SELU curve.
    :param alpha: a scalar parameter that influences the slope of the SELU curve.
    :return: the result of the derivative of SELU function applied to x.
    """
    return np.where(x > 0, lambd, lambd * alpha * np.exp(x))



activation_funcs = {
    'identity': Activation_function(identity, identity_derivative, 'Identity'),
    'sigmoid': Activation_function(sigmoid, sigmoid_derivative, 'Sigmoid'),
    'tanh': Activation_function(tanh, tanh_derivative, 'Tanh'),
    'relu': Activation_function(ReLU, ReLU_derivative, 'ReLU'),
    'leaky_relu': Activation_function(leaky_ReLU, leaky_ReLU_derivative, 'Leaky ReLU'),
    'selu': Activation_function(SELU, SELU_derivative, 'SELU')
}