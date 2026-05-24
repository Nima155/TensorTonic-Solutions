import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    states = []
    last = None
    for t in range(X.shape[1]):  
        print(h_0)
        h_0 = np.tanh(h_0 @ W_hh.T + X[:, t, :] @ W_xh.T + b_h)
        states.append(h_0) 
        last = h_0
        # return h_0.shape
    return (np.stack(states, axis=1), last)