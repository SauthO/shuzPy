if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from shuzPy import Variable, as_variable, Model
import shuzPy.functions as F
from shuzPy.models import MLP

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.get_item(x, 1)
print(y)
y.backward()
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
indeces = np.array([0,0,1])
y = F.get_item(x, indeces)
print(y)

y = y[:2]
print(y)


x = Variable(np.array([[0.1, -0.4]]))
model = MLP((10, 3))
y = model(x)
p = F.softmax(y)
print(y)
print(p)

