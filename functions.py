import numpy as np
from shuzPy.core import Function

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

class