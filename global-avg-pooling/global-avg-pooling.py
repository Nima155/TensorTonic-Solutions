import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x = np.array(x)
    if len(x.shape) > 3:
        return x.mean(axis=2).mean(axis=2)
    else:
        return x.mean(axis=1).mean(axis=1)        