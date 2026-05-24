import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """
    Returns: np.ndarray of shape (B, H, W, C_out) after sequential linear transforms with ReLU
    """
    # Your implementation here
    # print(x.shape, np.array(weights).shape)
    # out = x @ np.array(weights) + np.array(biases)
    # out[out < 0] = 0
    for w, b in zip(weights, biases):
        x = x @ np.array(w) + np.array(b)
        x[x < 0] = 0
    return x
    # return out