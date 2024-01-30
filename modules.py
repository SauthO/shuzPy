import numpy as np
import unittest
import weakref
from memory_profiler import profile
import contextlib

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
                
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
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
            #print(f.outputs[0].data)
            gys = [output().grad for output in f.outputs]
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
                    y().grad = None

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

Variable.__neg__ = neg    

class Add(Function):
    
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

Variable.__add__ = add
Variable.__radd__ = add

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

Variable.__sub__ = sub
Variable.__rsub__ = rsub

class Mul(Function):

    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return x1 * gy, x0 * gy

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

Variable.__mul__ = mul
Variable.__rmul__ = mul

class Div(Function):
    
    def forward(self, x0, x1):
        return x0/x1
    
    def backward(self, gy):
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return gy / x1 , ( -x0 / x1 ** 2 ) * gy

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()( x1, x0 ) 


Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv

class Pow(Function):
    
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        return c * x ** (c-1) * gy

def pow(x, c):
    return Pow(c)(x)

Variable.__pow__ = pow

class Square(Function):
    
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.inputs[0].data
        return 2*x*gy
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x)*gy

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

a = Variable(np.array(2.0))
b = Variable(np.array(3.0))
c = Variable(np.array(4.0))

y = 2*(-a)-2.0/c + b**2

y.backward()

print(y)
print(a.grad)
print(b.grad)
print(c.grad)
print(3.0/a)
print(a/3.0)