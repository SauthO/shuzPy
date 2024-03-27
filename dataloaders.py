import math
import numpy as np
import random



class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size/batch_size)
        
        self.reset()
    
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        #for x, t in loaderのイテレーションを回すたびにこの__next__が呼ばれる
        if self.iteration >= self.max_iter:
            self.reset() # このリセットでepochごとにindexがシャッフルされる。
            raise StopIteration
        
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size : (i+1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
    
    def next(self):
        return self.__next__()
    

"""
class MyIterator:
    
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()
        
        self.cnt += 1
        return self.cnt

obj = MyIterator(5)
print(next(obj))
print(next(obj))
print(next(obj))
print(next(obj))
print(next(obj))
"""
