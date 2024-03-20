import copy as cp
import math

import numpy as np

from layers.BaseLayer import BaseLayer
from layers.utils import pad_values, img_to_col, col_to_image


class Conv2D(BaseLayer):
    def __init__(self, number_of_filters, f_size, inp_size=None, stride=1, pad_val='same'):
        self.fil_count = number_of_filters
        self.padding = pad_val
        self.stride = stride
        self.f_size = f_size
        self.inp_size = inp_size

    # Initializing the values
    def initialize_value(self, optimizer):
        h_f, w_f = self.f_size
        val = 1 / math.sqrt(np.prod(self.f_size))
        channel = self.inp_size[0]
        self.w = np.random.uniform(-val, val, size=(self.fil_count, channel, h_f, w_f))
        self.w0 = np.zeros((self.fil_count, 1))
        self.wopt = cp.copy(optimizer)
        self.w0opt = cp.copy(optimizer)

    # Calculating the number of parameters
    def params(self):
        val1 = self.w0.shape
        val2 = self.w.shape
        return np.prod(val1) + np.prod(val2)

    # Defining forward flow of input values
    def front_flow(self, X, train=True):
        sizeofbatch, channel, ht, wt = X.shape
        self.in_lyr = X
        self.Wcol = self.w.reshape((self.fil_count, -1))
        self.Xcol = img_to_col(X, self.f_size, output=self.padding, stride=self.stride)
        o = self.Wcol.dot(self.Xcol) + self.w0
        o = o.reshape(self.get_output() + (sizeofbatch,))
        return o.transpose(3, 0, 1, 2)

    def get_output(self):
        c, ht, wt = self.inp_size
        h_p, w_p = pad_values(self.f_size, padding=self.padding)
        o_ht = (ht + np.sum(h_p) - self.f_size[0]) / self.stride + 1
        o_wt = (wt + np.sum(w_p) - self.f_size[1]) / self.stride + 1
        return self.fil_count, int(o_ht), int(o_wt)

    # Defining backward flow from output layer
    def back_flow(self, totalgrad):
        totalgrad = totalgrad.transpose(1, 2, 3, 0)
        totalgrad = totalgrad.reshape(self.fil_count, -1)
        grad_w = totalgrad.dot(self.Xcol.T).reshape(self.w.shape)
        grad_w0 = np.sum(totalgrad, keepdims=True, axis=1, )
        self.w = self.wopt.update(self.w, grad_w)
        self.w0 = self.w0opt.update(self.w0, grad_w0)
        totalgrad = self.Wcol.T.dot(totalgrad)
        totalgrad = col_to_image(totalgrad,
                                 self.in_lyr.shape,
                                 self.f_size,
                                 o_shape=self.padding,
                                 stride=self.stride,
                                 )

        return totalgrad
