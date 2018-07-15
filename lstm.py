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


def create_rand_array(a,b,*args):
    return np.random.rand(*args) * (b-a) +a


""" 
    Create class LSTMNode as main class
    LSTMParam stores the parameters required for each state
    LSTMState Class stores the state values.
    LSTMNetwork has functions to support creation of LSTM Networks


"""

