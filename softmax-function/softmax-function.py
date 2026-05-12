import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x)
    if len(x.shape) > 1:
        row_wise = np.sum(np.exp(x - np.max(x)), axis=1)
        return np.exp(x - np.max(x)) / row_wise.reshape((row_wise.shape[0], -1)) 
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))