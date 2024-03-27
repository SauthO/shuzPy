import numpy as np
import math

class Optimizer:
    
    def __init__(self):
        self.target = None
        self.hooks = []
    
    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        # gradがNoneでないパラメータをリストにまとめる
        params = [p for p in self.target.params() if p.grad is not None]
        
        # 前処理(オプション)
        for f in self.hooks:
            print(f)          

        # parameter の更新
        for param in params:
            self.update_one(param)
    
    def update_one(self, param):
        raise NotImplementedError()
    
    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer): #Stochastic Gradient Descent

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
    
    def update_one(self, param):
        param.data -= self.lr * param.grad.data
    
class MomentumSGD(Optimizer):
    
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}
    
    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs: #update_oneは初めて呼ばれるときにパラメータと同じ形状のデータを生成
            self.vs[v_key] = np.zeros_like(param.data)
        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):

    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = np.zeros_like(param.data)
        h = self.hs[h_key]
        h += param.grad.data * param.grad.data
        param.data -= self.lr * param.grad.data / ( np.sqrt(h) + self.eps )


class Adam(Optimizer):

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)
    
    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1


    def update_one(self, param):
        key = id(param)

        if key not in self.ms:
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)
        
        m, v = self.ms[key], self.vs[key]
        m += ( 1 - self.beta1 ) * ( param.grad.data - m)
        v += ( 1 - self.beta2 ) *( param.grad.data * param.grad.data - v )
        param.data -= self.lr * m / ( np.sqrt(v) + self.eps ) 