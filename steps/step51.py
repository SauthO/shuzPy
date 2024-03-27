if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))



import matplotlib.pyplot as plt
import numpy as np
from shuzPy import datasets
from shuzPy.dataloaders import DataLoader
from shuzPy.models import MLP
from shuzPy import optimizers
import shuzPy.functions as F 
import shuzPy

"""
train_set = shuzPy.datasets.MNIST(train=True, transform=None)
test_set = shuzPy.datasets.MNIST(train=False, transform=None)

print(len(train_set))
print(len(test_set))

print(type(train_set))
x,t = train_set[1]
print(type(x), x.shape)
print(t)


plt.imshow(x.reshape(28,28), cmap="gray")
plt.axis("off")
plt.show()
print(f"label:{t}")
"""

def f(x):
    x = x.flatten()
    x = x.astype(np.float)
    x /= 255.0
    return x

train_set = datasets.MNIST(train=True, transform=f)
test_set = datasets.MNIST(train=False, transform=f)

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_loader = DataLoader(train_set, batch_size) 
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)

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

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

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
plt.subplots_adjust(wspace=0.5, hspace=1) #グラフの間隔
plt.subplot(1, 2, 2)
plt.title("acc")
plt.plot(epoch_list, train_acc, label="train_acc")
plt.plot(epoch_list, test_acc, label="test_acc")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.ylim([0.0, 1.0])

plt.subplot(1, 2, 1)
plt.title("loss")
plt.plot(epoch_list, train_loss, label="train_loss")
plt.plot(epoch_list, test_loss, label="test_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

