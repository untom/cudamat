import os
import pdb
import platform
import warnings
import ctypes as ct
import numpy as np
from cudamat import generate_exception

if platform.system() == 'Windows':
    _cudalearn = ct.cdll.LoadLibrary('libcudalearn.dll')
else:
    _cudalearn = ct.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__) or os.path.curdir, 'libcudalearn.so'))

_cudalearn.mult_by_sigmoid_deriv.restype = ct.c_int
_cudalearn.calculate_l1_penalty.restype = ct.c_int

def mult_by_sigmoid_deriv(target, acts):
    """
    target = target * acts * (1 - acts)

    Useful for doing backprop in neural networks with logistic units.
    """

    err_code = _cudalearn.mult_by_sigmoid_deriv(target.p_mat, acts.p_mat)
    if err_code:
        raise generate_exception(err_code)

def calculate_l1_penalty(mat, alpha, target = None):
    """
    Calculates the L1 regularization penalty of size p to mat.
    
    Equivalent to:
        l1_penalty = alpha * np.sign(X)
        result = np.where(alpha > np.abs(X), X, l1_penalty) # don't overshoot 0
    """

    if not target:
        target = mat

    elif isinstance(alpha, (int, float)):
        err_code = _cudalearn.calculate_l1_penalty(mat.p_mat, ct.c_float(alpha), target.p_mat)
        if err_code:
            raise generate_exception(err_code)
    else:
        raise ValueError, "Value must be of type int or float."

    return target
