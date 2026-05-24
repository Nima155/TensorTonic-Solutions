import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True, mask: np.ndarray = None) -> np.ndarray:
    """
    Apply inverted dropout. If mask is provided, use it; otherwise generate one.
    """
    # YOUR CODE HERE
    if training:
        if mask is not None:
            x[mask == 1] *= (1 / (1-p))
            x[~(mask == 1)] = 0
            return x
    else:
        return x