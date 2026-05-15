import numpy as np
from collections import defaultdict
def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    df = defaultdict(int)
    mx, most_freq_label = 0, -1
    for y in y_train:
        df[y] += 1
        if df[y] > mx:
            mx = df[y]
            most_freq_label = y
    return np.zeros_like(X_test) + most_freq_label