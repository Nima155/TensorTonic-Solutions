import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    # YOUR CODE HERE
    # return image.shape
    
    return np.zeros((image.shape[0], 
        ((image.shape[1] - 11 + 4) // 4) + 1, 
        ((image.shape[2] - 11 + 4) // 4) + 1,
        96
                    ))