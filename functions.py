if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
from shuzPy.core import Function
from shuzPy.core import as_variable
from shuzPy import utils

class Exp(Function):

    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]() #weakref
        gx = y * gy
        return gx

def exp(x):
    return Exp()(x)

class Log(Function):

    def forward(self, x):
        y = np.log(x)
        return y
    
    def backward(self, gy):
        (x,) = self.inputs
        gx = gy / x
        return gx
    
def log(x):
    return Log()(x)

class Sin(Function):

    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        (x,) = self.inputs
        gx = cos(x) * gy
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):

    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        (x,) = self.inputs
        gx = -sin(x) * gy
        return gx
    #20240205
    #backwardでreturn -np.sin(x) * gy としていたのでエラーが出ていた
    #TypeError: loop of ufunc does not support argument 0 of type Variable which has no callable cos method

def cos(x):
    return Cos()(x)

class Tanh(Function):

    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        y = self.outputs[0]() #outputs is weakref
        gx = ( 1 - y**2 ) * gy
        return gx
    
def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape) #ndarrayのreshape
    
    def backward(self, gy):
        gx = reshape(gy, self.x_shape)
        return gx

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    
    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):
        gx = transpose(gy)
        return gx

def transpose(x):
    return Transpose()(x)

class Sum(Function):

    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    
    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims) #ndarrayのメソッド
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
    
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
    
class SumTo(Function):
        
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)
        
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    
    def forward(self, x, W):
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    
def matmul(x, W):
    return MatMul()(x,W)

class MeanSquareError(Function):
    
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = diff*(2./len(diff)) * gy
        gx1 = -gx0
        return gx0, gx1

def mean_square_error(x0, x1):
    return MeanSquareError()(x0, x1)

def linear_simple(x, W, b=None):    
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y

class Linear(Function):

    def forward(self, x, W, b):
        y = x.dot(W) # forwardはcallの中で呼ばれて引数はndarrayが渡される
        if b is not None:
            y += b
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T) # self.inputsはVariable
        gW = matmul(x.T, gy)
        return gx, gW, gb
    
def linear(x, W, b=None):
    return Linear()(x, W, b)

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y

class Sigmoid(Function):

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = y*(1 - y) * gy
        return gx

def sigmoid(x):
    return Sigmoid()(x)

class GetItem(Function):

    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        y = x[self.slices]
        return y
    
    def backward(self, gy):
        (x,) = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):

    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)

def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y

def softmax_cross_entropy_simple(x,t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax_simple(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class Softmax(Function):
    
    def __init__(self, axis=1):
        self.axis = axis
    
    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y
    
    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx

def softmax(x, axis=1):
    return Softmax(axis)(x)


# =============================================================================
# max / min / clip
# =============================================================================

class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


def min(x, axis=None, keepdims=False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_ bbbbbbbbbbbbbbbbbbbbbbbbfxcdcxddxddxmax):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)