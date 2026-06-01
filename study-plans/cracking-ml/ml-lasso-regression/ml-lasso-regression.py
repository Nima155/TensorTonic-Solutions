def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.array(X).astype(np.float64)
    y = np.array(y).astype(np.float64)
    b = 0.0
    w = np.array([0.0] * X.shape[1])
    N = X.shape[0]
    for _ in range(epochs):
        y_hat = (X @ w) + b
        err = y_hat - y
        w -= lr * (((2.0 / N) * (X.T @ err)) + alpha * np.sign(w))
        b -= lr * ((2.0 / N) * (err).sum())
    return (w, b)