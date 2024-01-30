import numpy as np
import unittest
import weakref

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
                
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
        
    def backward(self):
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

    def cleargrad(self):
        self.grad = None

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        self.generation = max([x.generation for x in inputs])
        for output in outputs: 
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs)>1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, gy):
        raise NotImplementedError()
    
class Add(Function):
    
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy
    
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
    
def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data)
print(x.grad)