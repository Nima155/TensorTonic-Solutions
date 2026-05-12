import numpy as np
# np.random.seed(123)/
def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)
    
    if rng:
        rnds = ((rng.random(x.shape) > p) * np.ones_like(x))/(1-p) 
    else:
        rnds = np.random.choice([0, 1/(1-p)], p=[p, 1-p], size=x.shape)
        
    
    return ((x * rnds), rnds)