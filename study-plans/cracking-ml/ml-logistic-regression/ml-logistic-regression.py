import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X)
    y = np.array(y)
    b = np.zeros(X.shape[0])
    w = np.zeros(X.shape[1])
    N = X.shape[0]
    for _ in range(n_iters):
        # print(X.shape, w.shape, b.shape)
        z = (X @ w) + b
        sigz = 1 / (1 + np.exp(-z))
        # mask = sigz >= 0.5
        # # sigz[mask] = 1
        # # sigz[~mask] = 0
        w -= lr * (1/N) * X.T @ (sigz - y)
        b -= lr * (1/N) * ((sigz - y).sum())
    return (w, b[0])