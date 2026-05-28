import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.array(C)
    row_sums = C.sum(axis = 1, keepdims=True)
    col_sums = C.sum(axis = 0, keepdims=True) 
    Es = (row_sums @ col_sums) / C.sum()
    # print()

    return ((((C - Es) ** 2) / Es).sum(), Es)