if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import numpy as np
from shuzPy import Function
from shuzPy import Variable
import math
from shuzPy.utils import plot_dot_graph
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001 # learning rate
iters = 10000
x0n = [0]*iters
x1n = [0]*iters

for i in range(iters):
    x0n[i] = float(x0.data)
    x1n[i] = float(x1.data)
    #print("(", x0n[i], ",", x1n[i], ")")
    y = rosenbrock(x0, x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -=  x0.grad * lr
    x1.data -=  x1.grad * lr

print(x0.data, x1.data)


plt.figure(figsize=(6,4),dpi=100)
plt.grid()
plt.rcParams["font.family"] = "DejaVu Serif"  

# 使用するフォント
plt.rcParams["font.size"] = 15                 # 文字の大きさ
plt.ylabel('y',fontsize=15)
plt.xlabel('x',fontsize=15)


x_, y_ = np.arange(-2.0,2.0,0.01), np.arange(-1.0,2.6,0.01)
x, y = np.meshgrid(x_, y_)
z = rosenbrock(x, y)

levs = 10**np.arange(0., 3.5, 0.5)

plt.plot(1.0, 1.0, zorder=10 ,marker="*", markersize=20, color="b")
plt.plot(x0n, x1n, "ro")
plt.contour(x,y,z,norm=LogNorm(),levels=levs,cmap="viridis" )
#plt.colorbar()
plt.title("rosenbrock")
plt.savefig("./img/rosenbrock_step28.pdf")
plt.show()

"""
x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

plot_dot_graph(y, verbose=False, to_file='./img/rosenbrock.pdf')
"""