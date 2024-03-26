if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import math
import numpy as np
import shuzPy
from shuzPy import optimizers 
from shuzPy import datasets
from shuzPy.models import MLP
import shuzPy.functions as F
from shuzPy import transforms
from shuzPy.datasets import Spiral
from shuzPy import DataLoader

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = shuzPy.datasets.Spiral(train=True)
test_set = shuzPy.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy_simple(y, t)
        acc = F.accuracy(y, t) 
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data)*len(t)
        sum_acc += float(acc.data)*len(t)
    print(f"epoch: {epoch+1:.4f}")
    print(f"train loss: {sum_loss/len(train_set):.4f}")
    print(f"train accuracy: {sum_acc/len(train_set):.4f}")
    print()


"""
batch_size = 10
max_epoch = 1

train_set = Spiral(train=True)
test_set = Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)
        print(t.data)
        break

    # check with test data after learning with train data
    for x, t in test_loader:
        print(x.shape, t.shape)
        break
"""