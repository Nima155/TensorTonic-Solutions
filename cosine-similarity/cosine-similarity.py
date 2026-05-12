import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a  = np.array(a)
    b = np.array(b)
    denominator = (np.linalg.norm(a) * np.linalg.norm(b))
    return (a @ b) / (denominator if denominator else 1)