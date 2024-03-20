import numpy as np

from layers.BaseLayer import BaseLayer


# ReLu Activation function definition
class ReluActivation:
    def gradient(self, val):
        return np.where(val < 0, 0, 1)

    def __call__(self, data):
        return np.where(data < 0, 0, data)


# Softmax Activation definition
class SoftmaxActivation:
    def gradient(self, data):
        p = self.__call__(data)
        return p * (1 - p)

    def __call__(self, data):
        ex = np.exp(data - np.max(data, keepdims=True, axis=-1))
        return ex / np.sum(ex,keepdims=True, axis=-1,)

# Defining Activation class
class Activation(BaseLayer):
    def __init__(self, func):
        self.lyr = None
        self.funct = func()

    def get_output(self):
        return self.inp_size

    def name(self):
        return "Activation ({})".format(self.funct.__class__.__name__)

    def front_flow(self, val, training=True):
        self.lyr = val
        return self.funct(val)

    def back_flow(self, total_gradient):
        return total_gradient * self.funct.gradient(self.lyr)

