import numpy as np
from copy import deepcopy
def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    
    # return pos
    ml = np.resize((2 * np.arange(0, d_model//2) / d_model), (seq_length, d_model//2))
    pe = np.sin(np.arange(0, seq_length).reshape((seq_length, 1))/(10000 ** ml))
    po = np.cos(np.arange(0, seq_length).reshape((seq_length, 1))/(10000 ** ml))

    pos = np.zeros((seq_length, d_model)).astype(np.float64)
    pos[:, ::2] = pe 
    pos[:, 1::2] = po 
    return pos