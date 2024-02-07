if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from shuzPy import Variable
import shuzPy.functions as F
from shuzPy.utils import plot_dot_graph
import matplotlib.pyplot as plt


# toy data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

plt.plot(x, y, "o")

x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x,W)+b
    return y

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W,b,loss)

plt.plot(x.data, y_pred.data)
plt.xlabel("x")
plt.ylabel("y")
plt.show()