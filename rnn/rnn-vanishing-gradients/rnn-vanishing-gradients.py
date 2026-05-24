import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    # YOUR CODE HERE
    ls = [1]
    W_hb = W_hh
    for t in range(T-1):
        ls.append(np.linalg.norm(W_hb, ord=2))
        W_hb = W_hh.T * np.linalg.norm(W_hb, ord=2)
    return ls