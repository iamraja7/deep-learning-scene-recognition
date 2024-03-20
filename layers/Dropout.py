import numpy as np

from layers.BaseLayer import BaseLayer


class Dropout(BaseLayer):
    def __init__(self, pval=0.2):
        self.pval = pval
        self.number_of_units = None
        self.pass_through = True
        self._mask_val = None
        self.inp_size = None

    # Defining output shape function
    def get_output(self):
        return self.inp_size

    # Defining forward flow
    def front_flow(self, X, train=True):
        cval = (1 - self.pval)
        if train:
            self._mask_val = np.random.uniform(size=X.shape) > self.pval
            cval = self._mask_val
        return X * cval

    # Defining backward flow
    def back_flow(self, total_gradient):
        return total_gradient * self._mask_val


