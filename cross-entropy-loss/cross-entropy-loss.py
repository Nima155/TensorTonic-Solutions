import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    arr = y_pred[list(range(0, y_pred.shape[0])), y_true]
    # print(arr)
    return -np.log(arr).mean()