if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import weakref
import numpy as np
import shuzPy.functions as F
from shuzPy.core import Parameter
#import os

class Layer:
    
    # layer = Layer()で呼ばれる
    def __init__(self):
        self._params = set()

    # layer.p = Parameter(np.array(1))などインスタンス変数を設定するたびに呼ばれる。
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)): # Layerも追加
            self._params.add(name)
        super().__setattr__(name, value)
    
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(input) for input in inputs]
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name] # layer.__dict__にはすべてのインスタンス変数が辞書型で格納されている

            if isinstance(obj, Layer): # 子Layerからパラメータを再帰的に取り出す。
                yield from obj.params() # ジェネレータを使って別のジェネレータを作る。
            
            else:
                yield obj
    """
    def params(self):
        yield self.__dict__["p1"]
        yield self.__dict__["p2"]
        ...
    params メソッドを呼ぶたびに上から順に返される。
    """
        
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

        """
        20240208
        for param in _params:
        とメモリ使用量か何かが得？
        比べる。
        """
    
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + "/" + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj
    
    def save_weights(self, path):
        #self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            # 保存途中にインタラプトがあると保存を中断しそのファイルを削除する
            # 不完全なファイルの作成を防ぐ
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        # out_sizeは必ず指定する。一つ目の層は手で入れる入力のサイズ、中間層は前の層のout_sizeを参考にin_sizeを決める

        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        self.W = Parameter(None, name="W")
        if self.in_size is not None: # in_size が設定されていないときは後回し
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")
    
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):

        # データを流すタイミングで重みを初期化
        if self.W.data is None:
            self.in_size = x.shape[1] #入力のサイズ参照
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y