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

