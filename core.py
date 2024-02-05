if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
import unittest
import weakref
#from memory_profiler import profile
import contextlib
import shuzPy

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        self.name = name

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'

    #enable narray instance variable
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return shuzPy.functions.reshape(self, shape)
    
    def transpose(self):
        return shuzPy.functions.transpose(self)
    
    @property
    def T(self):
        return shuzPy.functions.transpose(self)
    
    def sum(self, axis=None, keepdims=False):
        return shuzPy.functions.sum(self, axis, keepdims)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            #self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)
                
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs] # output is weakref
            #gys = [output.grad for output in f.outputs] 
            
            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else :
                        x.grad = x.grad + gx
                        #x.grad += gx
                    if x.creator is not None:
                        add_func(x.creator)
            
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None # y is weakref
                        #y.grad = None

    def cleargrad(self):
        self.grad = None

#begin: for backprop invalidation mode--------------------
class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)
# end: for backprop invalidation mode--------------------------------------------------

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(input) for input in inputs]
        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])
            for output in outputs: 
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            #self.outputs = [output for output in outputs]
        return outputs if len(outputs)>1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, gy):
        raise NotImplementedError()

class Neg(Function):
    
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
def neg(x):
    return Neg()(x)

class Add(Function):
    
    def forward(self, x0, x1):
        return x0 + x1
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Sub(Function):
    
    def forward(self, x0, x1):
        return x0 - x1
    
    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Mul(Function):

    def forward(self, x0, x1):
        return x0 * x1 
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return x1 * gy, x0 * gy

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Div(Function):
    
    def forward(self, x0, x1):
        return x0/x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy / x1 , ( -x0 / x1 ** 2 ) * gy

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()( x1, x0 ) 

class Pow(Function):
    
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        (x,) = self.inputs
        #x = self.inputs 
        # => TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'
        #2024/02/02
        """
        a=[1]
        print(a)
        print(type(a))
        b,=[1]
        print(b)
        print(type(b))
        """

        c = self.c
        return c * x ** (c-1) * gy

def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow