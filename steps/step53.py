if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import os
from shuzPy.layers import Layer
from shuzPy.core import Parameter
import shuzPy.functions as F
from shuzPy.models import MLP
from shuzPy import DataLoader
from shuzPy import optimizers
from shuzPy import datasets
import shuzPy

x = np.array([1, 2, 3])
np.save("test.npy", x)
x = np.load("test.npy")
print(x)

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
data = {"x1":x1, "x2":x2}

np.savez_compressed("test.npz", **data)
arrays = np.load("test.npz")
x1 = arrays["x1"]
x2 = arrays["x2"]
print(x1)
print(x2)

layer = Layer()
l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))

print(layer.l1.p1)

param_dict = {}
layer._flatten_params(param_dict)
print(param_dict)


max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
test_set = datasets.MNIST(train=False)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)

if not os.path.exists("my_mlp.npz"):
    
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(max_epoch):
        sum_loss, sum_acc = 0, 0
        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            model.cleargrads()
            loss.backward()
            acc = F.accuracy(y, t)
            optimizer.update()
            sum_loss += float(loss.data)*len(t)
            sum_acc += float(acc.data)*len(t)
        
        print("=======================")
        print(f"epoch: {epoch+1}")
        print("=======================")
        train_loss.append(sum_loss/len(train_set))
        train_acc.append(sum_acc/len(train_set))
        print(f"train loss: {sum_loss/len(train_set):.4f}")
        print(f"train accuracy: {sum_acc/len(train_set):.4f}")
    
        # epoch ごとにテストデータで認識制度をチェック
        sum_loss, sum_acc = 0, 0
        with shuzPy.no_grad(): #
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy_simple(y, t)
                acc = F.accuracy(y, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)
        print("-----------------------")
        test_loss.append(sum_loss/len(test_set))
        test_acc.append(sum_acc/len(test_set))
        print(f"test loss: {sum_loss/len(test_set):.4f}")
        print(f"test accuracy: {sum_acc/len(test_set):.4f}")
        print()

    model.save_weights("my_mlp.npz")

else:
    test_loss = []
    test_acc = []
    model.load_weights("my_mlp.npz")
    sum_loss, sum_acc = 0, 0
    with shuzPy.no_grad(): 
        for x, t in test_loader: # イテレーションを回すたびにtest_loaderの__next__が呼ばれる
            y = model(x)
            loss = F.softmax_cross_entropy_simple(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
            
        test_loss.append(sum_loss/len(test_set))
        test_acc.append(sum_acc/len(test_set))
        print(f"test loss: {sum_loss/len(test_set):.4f}")
        print(f"test accuracy: {sum_acc/len(test_set):.4f}")
        print()