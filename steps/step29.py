if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import numpy as np
from shuzPy import Function
from shuzPy import Variable
import math
from shuzPy.utils import plot_dot_graph
import matplotlib.pyplot as plt

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        return np.cos(x) * gy

def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.000000001):
    y = 0

    for i in range(100000):
        c = (-1)**i / math.factorial(2*i + 1)
        t = c * x**(2*i+1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

def rosenbrock(x0, x1):
    y = 100*(x1-x0**2)**2 + (1-x0)**2
    return y

def f(x):
    return x**4 - 2*x**2

def fx2(x):
    return 13*x**2 - 4

X = 0.5
x = Variable(np.array(X))

# Newton

iters_fx = 10
x0n = [None]*iters_fx
y0n = [None]*iters_fx

for i in range(iters_fx):
    y = f(x)
    x0n[i] = float(x.data)
    y0n[i] = float(y.data)
    x.cleargrad()
    y.backward()
    x.data -= x.grad / fx2(x.data)    


# gradient descent
    
iters = 100
lr = 0.01
x1n = [0]*iters
y1n = [0]*iters
x = Variable(np.array(X))

for i in range(iters):
    x1n[i] = float(x.data)
    y = f(x)
    y1n[i] = float(y.data)
    x.cleargrad()
    y.backward()

    x.data -=  x.grad * lr


plt.figure(figsize=(6,4),dpi=100)
plt.grid()
plt.rcParams["font.family"] = "DejaVu Serif"  

# 使用するフォント
plt.rcParams["font.size"] = 15                 # 文字の大きさ
plt.ylabel('y',fontsize=15)
plt.xlabel('x',fontsize=15)

plt.plot(1.0, -1.0 , zorder=10 ,marker="*", markersize=20, color="b")
a = np.arange(-2.0, 2.0, 0.1)
b = f(a)

plt.plot(x0n, y0n, "ro", zorder=10,  linestyle='solid', label="newton")
plt.plot(x1n, y1n, "o" , linestyle='solid', label= "gradient descent")
plt.plot(a, b)
plt.legend()
plt.title("gradient descent vs newton")
plt.savefig("./img/newton_step29.pdf")
plt.show()

"""
x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

plot_dot_graph(y, verbose=False, to_file='./img/rosenbrock.pdf')
"""