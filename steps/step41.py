if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from shuzPy import Variable
import shuzPy.functions as F
from shuzPy.utils import plot_dot_graph

x = Variable(np.random.rand(2,3))
W = Variable(np.random.rand(3,4))
y = F.matmul(x, W)
print(y)
y.backward()
print(x.grad.shape)
print(W.grad.shape)