if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import shuzPy.datasets
import matplotlib.pyplot as plt

x, t = shuzPy.datasets.get_spiral(train=True)

print(x.shape)
print(t.shape)

print(x[10], t[10])
print(x[110], t[110])

x0 = []
y0 = []
x1 = [] 
y1 = [] 
x2 = [] 
y2 = []

for i in range(x.shape[0]):
    if t[i] == 0:
        x0.append(x[i][0])
        y0.append(x[i][1])
    elif t[i] == 1:
        x1.append(x[i][0])
        y1.append(x[i][1])
    elif t[i] == 2:
        x2.append(x[i][0])
        y2.append(x[i][1])

plt.plot(x0, y0 , marker="o", linestyle="None", color="orange")
plt.plot(x1, y1 , marker="x", linestyle="None", color="mediumblue")
plt.plot(x2, y2 , marker="^", linestyle="None", color="green")
plt.show()