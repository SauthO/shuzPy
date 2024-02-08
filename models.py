if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


from shuzPy import Layer
from shuzPy import utils
import shuzPy.functions as F
import shuzPy.layers as L

class Model(Layer):
    def plot(self, *inputs, to_file="model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, "l"+str(i), layer)
            self.layers.append(layer)
        
    def forward(self, x):
        for l in self.layers[:-1]:# 一番最後のlayerの一つ前まで
            x = self.activation(l(x))
        return self.layers[-1](x)