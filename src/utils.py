import numpy as np


def gain_function(x, a, b, d):
    """
    Args:
    x (np.array):    input
    a (float):       function gain
    b (float):       function threshold
    d (float):       function noise factor
    
    Returns:
    y (np.array):    output
    """
    return (a * x - b) / (1 - np.exp(-d * (a * x - b)))
