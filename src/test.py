from functions import loss_functions
from functions import activation_functions
from functions import regularization_functions
import numpy as np

def increment(a,b,c,d):
    return a+1,b+1,c+1,d+1

a,b,c,d = 0,0,0,0

a,b,c,d += increment(a,b,c,d)
