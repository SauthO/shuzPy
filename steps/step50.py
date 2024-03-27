if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import math
import numpy as np
import matplotlib.pyplot as plt
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

train_loss = []
train_acc = []
test_loss = []
test_acc = []

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
        #print(sum_loss)
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


epoch_list = [i for i in range(max_epoch)]

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 2)
plt.title("acc")
plt.plot(epoch_list, train_acc, label="train_acc")
plt.plot(epoch_list, test_acc, label="test_acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
#plt.ylim([0.0, 1.0])

plt.subplot(1, 2, 1)
plt.title("loss")
plt.plot(epoch_list, train_loss, label="train_loss")
plt.plot(epoch_list, test_loss, label="test_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()
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