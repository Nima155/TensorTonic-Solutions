import numpy as np
from collections import defaultdict
def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    df = defaultdict(int)
    for x in y:
        df[x] += 1

    counts = np.array(list(df.values()))
    p_i = counts / counts.sum()
    return -(p_i * np.log2(p_i)).sum()