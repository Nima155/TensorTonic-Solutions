import numpy as np
def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    if not set_a and not set_b:
        return 0
    set_a = np.array(set_a)
    set_b = np.array(set_b)
    return (np.intersect1d(set_a, set_b)).size / (np.union1d(set_a, set_b).size)