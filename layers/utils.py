import math
from datetime import datetime

import numpy as np

SAME_PADDING = "same"
VALID_PADDING = "valid"



def diagonal_matrix(x):
    mat = np.zeros((len(x), len(x)))
    for j in range(len(mat[0])):
        mat[j, j] = x[j]
    return mat

def iter_batch(X, y=None, batch_size=64):
    number_samp = X.shape[0]
    for i in np.arange(0, number_samp, batch_size):
        start, end = i, min(i + batch_size, number_samp)
        if y is not None:
            yield X[start:end], y[start:end]
        else:
            yield X[start:end]

def normalize(X, order=2, axis=-1):
    lval2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    lval2[lval2 == 0] = 1
    return X / np.expand_dims(lval2, axis)


def pad_values(filter, padding=SAME_PADDING):
    if padding == "valid":
        return (0, 0), (0, 0)
    elif padding == SAME_PADDING:
        h_filter, w_filter = filter
        h1 = int(math.floor((h_filter - 1) / 2))
        w1 = int(math.floor((w_filter - 1) / 2))
        h2 = int(math.ceil((h_filter - 1) / 2))
        w2 = int(math.ceil((w_filter - 1) / 2))
        return (h1, h2), (w1, w2)


def find_column_values(image, filter, padding, stride=1):
    batch_size, channel, ht, wt = image
    h_f, w_f = filter
    h_p, w_p = padding
    hout = int((ht + np.sum(h_p) - h_f) / stride + 1)
    wout = int((wt + np.sum(w_p) - w_f) / stride + 1)

    a0 = np.repeat(np.arange(h_f), w_f)
    a0 = np.tile(a0, channel)
    a1 = stride * np.repeat(np.arange(hout), wout)
    b0 = np.tile(np.arange(w_f), h_f * channel)
    b1 = stride * np.tile(np.arange(wout), hout)
    a = a0.reshape(-1, 1) + a1.reshape(1, -1)
    b = b0.reshape(-1, 1) + b1.reshape(1, -1)

    l = np.repeat(np.arange(channel), h_f * w_f).reshape(-1, 1)
    return (l, a, b)


def img_to_col(imgs, filter, stride, output=SAME_PADDING):
    h_p, w_p = pad_values(filter, output)
    img_pad = np.pad(imgs, ((0, 0), (0, 0), h_p, w_p), mode='constant')
    k, i, j = find_column_values(imgs.shape, filter, (h_p, w_p), stride)
    columns = img_pad[:, k, i, j]
    channel = imgs.shape[1]
    f_h, f_w = filter
    columns = columns.transpose(1, 2, 0).reshape(f_h * f_w * channel, -1)
    return columns


def col_to_image(columns, img_shape, filter, stride, o_shape=SAME_PADDING):
    b_size, channel, ht, wt = img_shape
    h_p, w_p = pad_values(filter, o_shape)
    h_padded = ht + np.sum(h_p)
    w_padded = wt + np.sum(w_p)
    ipadval = np.zeros((b_size, channel, h_padded, w_padded))
    l, a, b = find_column_values(img_shape, filter, (h_p, w_p), stride)
    columns = columns.reshape(channel * np.prod(filter), -1, b_size)
    columns = columns.transpose(2, 0, 1)
    np.add.at(ipadval, (slice(None), l, a, b), columns)
    return ipadval[:, :, h_p[0]:ht + h_p[0], w_p[0]:wt + w_p[0]]


class AdamOptimizer:
    def __init__(self, rate=0.001, decay_rate1=0.9, decay_rate2=0.999):
        self.delta = None
        self.rate = rate
        self.eps = 1e-8
        self.momentum = None
        self.velocity = None
        self.decay_rate1 = decay_rate1
        self.decay_rate2 = decay_rate2

    def update(self, original_weight, weight_grad):
        if self.momentum is None:
            self.momentum = np.zeros(np.shape(weight_grad))
            self.velocity = np.zeros(np.shape(weight_grad))

        self.momentum = self.decay_rate1 * self.momentum + (1 - self.decay_rate1) * weight_grad
        self.velocity = self.decay_rate2 * self.velocity + (1 - self.decay_rate2) * np.power(weight_grad, 2)

        updated_velocity = self.velocity / (1 - self.decay_rate2)
        updated_momentum = self.momentum / (1 - self.decay_rate1)

        self.delta = self.rate * updated_momentum / (np.sqrt(updated_velocity) + self.eps)
        return original_weight - self.delta


def acc_score(y_true, y_pred):
    return np.sum(y_pred == y_true, axis=0) / len(y_true)


class Loss(object):
    def loss(self, y_actual, y_predict):
        pass

    def gradient(self, y_actual, y_predict):
        pass

    def calculate_accuracy(self, y_actual, y_predict):
        return 0


class SquaredLoss(Loss):
    def __init__(self): pass

    def loss(self, y_actual, y_predict):
        return 0.5 * np.power((y_actual - y_predict), 2)

    def gradient(self, y_actual, y_pred):
        return -(y_actual - y_pred)


class CalCrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, pr):
        # Clipping probability to avoid divide by zero error
        pr = np.clip(pr, 1e-15, 1 - 1e-15)
        return - y * np.log(pr) - (1 - y) * np.log(1 - pr)

    def calculate_accuracy(self, y, pr):
        return acc_score(np.argmax(y, axis=1), np.argmax(pr, axis=1))

    def gradient(self, y, pr):
        # Clipping probability to avoid divide by zero error
        pr = np.clip(pr, 1e-15, 1 - 1e-15)
        return - (y / pr) + (1 - y) / (1 - pr)


def get_time_diff(start_time):
    return str((datetime.now() - start_time)).split(".")[0]
