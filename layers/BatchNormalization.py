import copy as cp

import numpy as np

from layers.BaseLayer import BaseLayer


class BatchNormalization(BaseLayer):
    def __init__(self, m=0.99, axis=0):
        self.m_run = None
        self.var_run = None
        self.m = m
        self.axis = axis
        self.ep = 0.01

    #Initializing default values
    def initialize_value(self, optimizer):
        self.scale = np.ones(self.inp_size)
        self.scale_optimizer = cp.copy(optimizer)
        self.offset_optimizer = cp.copy(optimizer)
        self.offset = np.zeros(self.inp_size)

    #Defining backward flow for Batch Normalization
    def back_flow(self, total_gradient):
        scale = self.scale
        X_normal = self.X_bar * self.inverse_stddev

        grad_offset = np.sum(total_gradient, self.axis)
        grad_scale = np.sum(total_gradient * X_normal, self.axis)

        self.offset = self.offset_optimizer.update(self.offset, grad_offset)
        self.scale = self.scale_optimizer.update(self.scale, grad_scale)

        b_size = total_gradient.shape[0]

        total_gradient = (1 / b_size) * scale * self.inverse_stddev * (
                b_size * total_gradient
                - self.X_bar * self.inverse_stddev ** 2 * np.sum(total_gradient * self.X_bar, self.axis)
                - np.sum(total_gradient, self.axis)
        )

        return total_gradient

    #Defining forward flow for Batch Normalization
    def front_flow(self, X, training=True):
        if self.m_run is None:
            self.var_run = np.var(X, self.axis)
            self.m_run = np.mean(X, self.axis)

        v_val = np.var(X, self.axis)
        m_val = np.mean(X, self.axis)

        self.var_run = self.m * self.var_run + (1 - self.m) * v_val
        self.m_run = self.m * self.m_run + (1 - self.m) * m_val

        self.inverse_stddev = 1 / np.sqrt(v_val + self.ep)
        self.X_bar = X - m_val
        
        normalized_X = self.X_bar * self.inverse_stddev
        output = self.scale * normalized_X + self.offset

        return output

    #Calculating the number of parameters
    def params(self):
        val1=self.offset.shape
        val2=self.scale.shape
        return np.prod(val1) + np.prod(val2)



    def get_output(self):
        return self.inp_size
