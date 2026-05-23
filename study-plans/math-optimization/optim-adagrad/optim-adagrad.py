import numpy as np
import math

def adagrad(X, y, lr, n_epochs):
    """
    Returns: tuple of (losses, effective_lrs) per epoch
    """
    X = np.array(X).astype(np.float64)
    y = np.array(y).astype(np.float64)
    w = np.zeros(X.shape[1]).astype(np.float64)
    G = np.zeros(X.shape[1]).astype(np.float64)
    N = X.shape[0]
    elr = lr / np.sqrt(G + 10e-8)
    losses = []
    elrs = []
    for i in range(n_epochs):
        losses.append(np.mean((X @ w - y) ** 2))
        g = (2 / N) * X.T @ (X @ w - y)
        G += g ** 2     
        elr = lr / np.array([math.sqrt(v + 10e-9) for v in G])
        elrs.append(elr)
        w -= elr * g
    return tuple([losses, elrs])
    # return w - lr * g