if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import shuzPy as shu
import unittest

def numerical_diff_x(f, x, y, eps=1e-4):
    x0 = shu.Variable(np.array(x.data-eps))
    x1 = shu.Variable(np.array(x.data+eps))
    z0 = f(x0, y)
    z1 = f(x1, y)

    return (z1.data-z0.data)/(2*eps)

def numerical_diff_y(f, x, y, eps=1e-4):
    y0 = shu.Variable(np.array(y.data-eps))
    y1 = shu.Variable(np.array(y.data+eps))
    z0 = f(x, y0)
    z1 = f(x, y1)

    return (z1.data-z0.data)/(2*eps)

def sphere(x, y):
    return x**2 + y**2

def matyas(x, y):
    return 0.26 * ( x**2 + y**2 ) - 0.48*x*y

def goldsteinprice(x, y):
    return ( 1 + ( x + y + 1 )**2 * ( 19 - 14*x + 3*x** 2 - 14*y +  6*x*y + 3*y**2) ) * \
            ( 20 + ( 2*x - 3*y )**2  * ( 18 - 32*x + 12*x**2 +48*y -36*x*y + 27*y**2 ))

def testfunc(x, y):
    return x*y + x**2 + y**2 - x/y - y/x

class DiffTest(unittest.TestCase):
    def test_gradient_check(self):
        x = shu.Variable(np.random.rand(1))
        y = shu.Variable(np.random.rand(1))
        z = testfunc(x, y)
        z.backward()
        num_grad_x = numerical_diff_x(testfunc, x, y)
        num_grad_y = numerical_diff_y(testfunc, x, y)
        flg_x = np.allclose(x.grad, num_grad_x)
        flg_y = np.allclose(y.grad, num_grad_y)
        self.assertTrue(flg_x)
        self.assertTrue(flg_y)

unittest.main()