import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.array(X).astype(np.float64)
    X = X - X.mean(axis = 0)
    sigma = 1/(X.shape[0] - 1) * X.T @ X

    std = np.sqrt(np.linalg.diagonal(sigma)).reshape((X.shape[1], -1))
    return sigma / (std @ std.T)