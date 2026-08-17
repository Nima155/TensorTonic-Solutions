import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.array(x)
    return (x.mean() - mu0) / (np.std(x, ddof=1) / np.sqrt(x.shape[0]))