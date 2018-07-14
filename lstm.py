import random

import numpy as np 
import math

def sigmoid(x):
    return 1./(1+np.exp(-x))

def sigmoid_derivative(values):
    """
    derivative of sigmoid(x) = x *(1-x)
    """
    return values * (1- values)  

def tanh_derivative ( values):
    return 1.- values**2