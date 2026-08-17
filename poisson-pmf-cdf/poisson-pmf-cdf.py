import numpy as np
import math
def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    return [((lam ** k) * np.exp(-lam)) / math.factorial(k), sum(((lam ** i) * np.exp(-lam)) / math.factorial(i) for i in range(k + 1))]