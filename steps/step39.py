if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from shuzPy import Variable
import shuzPy.functions as F
from shuzPy.utils import plot_dot_graph


x0 = Variable(np.array([1,2,3]))
x1 = Variable(np.array([10]))
print(x0+x1)

x = Variable(np.array([1,2,3,4,5,6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)