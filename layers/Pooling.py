import numpy as np

from layers.BaseLayer import BaseLayer
from layers.utils import VALID_PADDING, img_to_col, col_to_image

# Defining pool layer class
class PoolingLayer(BaseLayer):
    def __init__(self, pool_shape_size=(2, 2), stride=2, padding=VALID_PADDING):
        self.shape_pool = pool_shape_size
        self.stride = pool_shape_size[0] if stride is None else stride
        self.pad_val = padding

    def front_flow(self, X, train=True):
        self.lyr_inp = X
        batchsize, channel, h, w = X.shape
        X = X.reshape(batchsize * channel, 1, h, w)
        _, h_out, w_out = self.get_output()
        X_col = img_to_col(X, self.shape_pool, self.stride, self.pad_val)
        out = self._pool_forward(X_col)
        out = out.reshape(h_out, w_out, batchsize, channel)
        out = out.transpose(2, 3, 0, 1)
        return out

    def back_flow(self, totalgrad):
        b_size, _, _, _ = totalgrad.shape
        channels, h, w = self.inp_size
        totalgrad = totalgrad.transpose(2, 3, 0, 1).ravel()
        total_gradient_col = self._pool_backward(totalgrad)
        totalgrad = col_to_image(total_gradient_col, (b_size * channels, 1, h, w), self.shape_pool,
                                 self.stride, self.pad_val)
        totalgrad = totalgrad.reshape((b_size,) + self.inp_size)
        return totalgrad

    def get_output(self):
        channel, h, w = self.inp_size
        h_out = (h - self.shape_pool[0]) // self.stride + 1
        w_out = (w - self.shape_pool[1]) // self.stride + 1
        return channel, int(h_out), int(w_out)

# Defining maxpool layer class
class MaxPooling2D(PoolingLayer):
    def _pool_forward(self, X_col):
        argument_maximum = np.argmax(X_col, axis=0).flatten()
        o = X_col[argument_maximum, range(argument_maximum.size)]
        self.cache = argument_maximum
        return o

    def _pool_backward(self, total_gradient):
        totalgradcol = np.zeros((np.prod(self.shape_pool), total_gradient.size))
        argument_maximum = self.cache
        totalgradcol[argument_maximum, range(total_gradient.size)] = total_gradient
        return totalgradcol
