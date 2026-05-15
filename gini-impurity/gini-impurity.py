import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    y_left = np.array(y_left)
    y_right = np.array(y_right)
    y_l = np.array([(y_left == c).sum() / y_left.size for c in np.unique(y_left)])     
    y_r = np.array([(y_right == c).sum() / y_right.size for c in np.unique(y_right)])
    tots = y_right.size + y_left.size
    # print(tots)
    if not tots:
        return 0.0
    return (y_left.size / tots) * (1 - (y_l ** 2).sum()) + (y_right.size / tots) * (1 - (y_r ** 2).sum())