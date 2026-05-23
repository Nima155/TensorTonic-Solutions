import numpy as np

def rmsprop(X, y, lr, decay, n_epochs):
    """
    Returns: tuple of (losses, effective_lrs) per epoch
    """
    X = np.array(X)
    y = np.array(y)
    E = np.zeros(X.shape[1])
    elr = lr / (np.sqrt(E + 10e-9))
    N = X.shape[0]
    w = np.zeros(X.shape[1])
    losses = []
    elrs = []
    for _ in range(n_epochs):
        losses.append(np.mean((X @ w - y) ** 2))
        gt = 2 / N * X.T @ ( X @ w - y )
        E = decay * E + (1-decay) * gt * gt
        elr = lr / (np.sqrt(E + 10e-9))
        elrs.append(elr)
        w -= elr * gt
    return tuple([losses, elrs])
        