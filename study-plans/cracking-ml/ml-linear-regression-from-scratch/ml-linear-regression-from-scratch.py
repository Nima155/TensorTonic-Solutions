import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X)
    y = np.array(y)
    b = np.zeros(X.shape[0])
    w = np.zeros(X.shape[1])
    N = X.shape[0]
    # print(b)
    for _ in range(epochs):
        y_hat = X @ w + b
        w -= lr * ((2 / N) * X.T @ (y_hat - y))
        b -= lr * ((2 / N) * (y_hat - y).sum())
    return (w, b[0])