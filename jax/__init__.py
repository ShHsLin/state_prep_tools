"""
Minimal JAX stub for testing purposes
"""
import numpy as np
import scipy.linalg

class Config:
    def update(self, key, value):
        pass

config = Config()

def grad(func):
    """Minimal gradient stub - just returns the function"""
    def wrapper(*args, **kwargs):
        return np.zeros(len(args[0]))  # Return zeros for gradient
    return wrapper

def jit(func):
    """Minimal JIT stub - just returns the function unchanged"""
    return func