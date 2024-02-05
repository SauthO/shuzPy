if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
from shuzPy.core import Function
from shuzPy.core import as_variable
from shuzPy import utils

class Sin(Function):

    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        (x,) = self.inputs
        return cos(x) * gy

def sin(x):
    return Sin()(x)

class Cos(Function):

    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        (x,) = self.inputs
        return -sin(x) * gy
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
        return ( 1 - y**2 ) * gy
    
def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape) #ndarrayのreshape
    
    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

class Transpose(Function):
    
    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):
        return transpose(gy)

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
        return broadcast_to(gy, self.x_shape)
    
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)
    
    def backward(self, gy):
        return sum_to(gy, self.x_shape)
    
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
        return broadcast_to(gy, self.x_shape)
    
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)