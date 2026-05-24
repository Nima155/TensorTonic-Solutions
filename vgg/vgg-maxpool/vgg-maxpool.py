import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    # print(x.shape)
    # print(x)
    mx = []
    for i in range(0, x.shape[1], 2):
        for j in range(0, x.shape[2], 2):
            mx.append(x[:,i:2+i,j:2+j, :].max(axis = (1, 2)))

    return np.array(mx).reshape((x.shape[0], x.shape[1]//2, x.shape[2]// 2, x.shape[-1]))