import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, num_classes) with classification logits
    """
    # Your implementation here
    x = features.reshape((features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))
    h1 = x @ W1 + b1
    h1[h1 < 0] = 0
    h2 = h1 @ W2 + b2 
    h2[h2 < 0] = 0
    y = h2 @ W3 + b3 
    return y