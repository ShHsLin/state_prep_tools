"""
Minimal JAX stub for testing purposes
"""
import numpy as np
import scipy.linalg

class Config:
    def update(self, key, value):
        pass

config = Config()

def grad(func, argnum=0):
    """Minimal gradient stub - returns a function that computes fake gradients"""
    def grad_func(*args, **kwargs):
        # Return zeros with appropriate shape for gradient
        if argnum < len(args):
            param_shape = np.array(args[argnum]).shape
            return np.zeros(param_shape)
        else:
            return np.zeros(10)  # Default size
    return grad_func

def jit(func):
    """Minimal JIT stub - just returns the function unchanged"""
    return func