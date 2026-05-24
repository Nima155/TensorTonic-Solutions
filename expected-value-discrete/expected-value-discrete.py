import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if np.array(p).sum() != 1:
        raise ValueError()
    return np.array(x) @ np.array(p)
