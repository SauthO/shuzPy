if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from shuzPy import Variable, Model
from shuzPy.layers import Layer
import shuzPy.layers as L
import shuzPy.functions as F
import matplotlib.pyplot as plt
