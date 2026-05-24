import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Apply 2D max pooling (shape simulation).
    """
    # YOUR CODE HERE
    return np.zeros(
        (
            x.shape[0],
            ((x.shape[1] - kernel_size) // stride) + 1,
            ((x.shape[2] - kernel_size) // stride) + 1,
            x.shape[-1]
        )
    )