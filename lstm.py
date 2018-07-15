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

# LSTMParam Class Implementation
""" 
mm_cell_count : LSTM cell count
x_dim = T_x  , the length of the input string

"""

class LSTMParam:
    def __init__(self,mm_cell_count,x_dim):
        self.mm_cell_count = mm_cell
        self.x_dim   = x_dim
        """ 
        The weights used in LSTM are made from 
        concatentaing the input text and activation values
        W_ax =[a<i> : x<i>]

        concat_length = size of cells + size of input (x)
        """
        concat_length  = x_dim + mm_cell

        """ 
        Create matrises for storing different weights

        """
        # output gate matrix
        self.wo = create_rand_array(-0.01,0.01,mm_cell_count,concat_length)
        self.wf = create_rand_array(-0.01,0.01,mm_cell_count,concat_length)
        self.wi = create_rand_array(-0.01,0.01,mm_cell_count,concat_length)
