import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    # print(np.pad(tpr[1:], 1))
    # print(tpr[:] + np.pad(tpr[1:], 1)[1:])
    # print(tpr[:] + np.pad(tpr[1:], 1)[1:])
    return (((1/2) * ((tpr[:] + np.pad(tpr[1:], 1)[1:]))[:-1]) * ((np.pad(fpr[1:], 1)[1:] - fpr))[:-1]).sum()