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


train_set = datasets.Spiral(train=True)
print(train_set[0])
print(len(train_set))

batch_index = [0, 1, 2]
batch = [train_set[i] for i in batch_index]
print(batch)


#前処理の例
def f(x):
    return x/2.0

train_set = datasets.Spiral(train=True, transform=f)
batch_index = [0, 1, 2]
batch = [train_set[i] for i in batch_index]
print(batch)



x = np.array([example[0] for example in batch])
t = np.array([example[1] for example in batch])

print(x)
print(t)

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = datasets.Spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size/batch_size)

for epoch in range(max_epoch):

    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i*batch_size : (i+1)*batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])


        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print(f"epoch {epoch+1}, loss {avg_loss:.2f}")