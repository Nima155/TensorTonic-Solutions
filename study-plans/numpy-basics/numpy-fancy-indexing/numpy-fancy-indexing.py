import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    arr = np.array(arr)
    return np.take(arr, indices, axis=axis).astype(np.float64)