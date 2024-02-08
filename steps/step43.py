if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from shuzPy import Variable
import shuzPy.functions as F
from shuzPy.utils import plot_dot_graph
import matplotlib.pyplot as plt

# data set
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2*np.pi*x) + np.random.rand(100, 1)
# init weight and bias 
I, H1, O = 1, 10, 1 # In, Hidden, Out
W1 = Variable(0.01 * np.random.randn(I, H1)) # gaus distribution
b1 = Variable(np.zeros(H1))
W2 = Variable(0.01 * np.random.randn(H1, O))
b2 = Variable(np.zeros(O))
#W3 = Variable(0.01 * np.random.randn(H2, O))
#b3 = Variable(np.zeros(O))


# predict of NN
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    #y = F.sigmoid(y)
    #y = F.linear(y, W3, b3)
    return y

lr = 0.2
iters = 10000

# learning of NN
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    #W3.cleargrad()
    #b3.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    #W3.data -= lr * W3.grad.data
    #b3.data -= lr * b3.grad.data

    if i%1000 == 0:
        print(loss)

# Plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()