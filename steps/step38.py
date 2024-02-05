if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from shuzPy import Variable
import shuzPy.functions as F
from shuzPy.utils import plot_dot_graph

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)
x = Variable(np.random.rand(2,3))
print(x)
y=x.reshape([3,2])
print(y)

print("transpose")
x = Variable(np.array([[1,2,3],[4,5,6]]))
print(x)
y = F.transpose(x)
print(y)
y.backward()
print(x.grad)

print("transpose method")
x = Variable(np.random.rand(2,3))
print(x)
y = x.transpose()
print(y)
y = y.T
print(y)