if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from shuzPy import Variable
from shuzPy.utils import plot_dot_graph

def goldsteinprice(x, y):
    return ( 1 + ( x + y + 1 )**2 * ( 19 - 14*x + 3*x** 2 - 14*y +  6*x*y + 3*y**2) ) * \
            ( 20 + ( 2*x - 3*y )**2  * ( 18 - 32*x + 12*x**2 +48*y -36*x*y + 27*y**2 ))

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))

y = goldsteinprice(x0 , x1)

y.backward()
print("grad", x0.grad)

x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

print(type(y))
print(id(y))
plot_dot_graph(y, verbose=False, to_file='./img/goldstein.pdf')