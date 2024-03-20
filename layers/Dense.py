import copy as cp
import math

import numpy as np

from layers.BaseLayer import BaseLayer


class Dense(BaseLayer):
    def __init__(self, number_of_units, inp_size=None):
        self.lyr_inp = None
        self.w = None
        self.w0 = None
        self.number_of_units = number_of_units
        self.inp_size = inp_size

    # Calculating the total number of parameters
    def params(self):
        return np.prod(self.w.shape) + np.prod(self.w0.shape)

    # Initializing values
    def initialize_value(self, optimizer):
        val = 1 / math.sqrt(self.inp_size[0])
        self.w = np.random.uniform(-val, val, (self.inp_size[0], self.number_of_units))
        self.w0 = np.zeros((1, self.number_of_units))
        self.wopt = cp.copy(optimizer)
        self.w0opt = cp.copy(optimizer)

    # Defining the forward flow function
    def front_flow(self, X, training=True):
        self.lyr_inp = X
        return self.w0 + X.dot(self.w)

    # Defining the backward flow function
    def back_flow(self, totalgrad):
        W = self.w
        grad_w = self.lyr_inp.T.dot(totalgrad)
        grad_w0 = np.sum(totalgrad, axis=0, keepdims=True)
        self.w = self.wopt.update(self.w, grad_w)
        self.w0 = self.w0opt.update(self.w0, grad_w0)
        totalgrad = totalgrad.dot(W.T)
        return totalgrad

    # Returning the output units
    def get_output(self):
        return (self.number_of_units,)
