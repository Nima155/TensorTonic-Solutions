def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.array(X).astype(np.float64)
    y = np.array(y).astype(np.float64)
    b = 0.0
    w = np.array([0.0] * X.shape[1])
    N = X.shape[0]
    for _ in range(epochs):
        y_hat = (X @ w) + b
        err = y_hat - y
        w -= lr * ((2.0 / N) * (X.T @ err) + 2.0 * alpha * w)
        b -= lr * ((2.0 / N) * (err).sum())
    return (w, b)