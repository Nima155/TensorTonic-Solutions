import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here

    dk = W_q.shape[0] // num_heads

    Q_t = (Q @ W_q).reshape((Q.shape[0], Q.shape[1], num_heads, dk)).transpose((0, 2, 1, 3))
    K_t = (K @ W_k).reshape((K.shape[0], K.shape[1], num_heads, dk)).transpose((0, 2, 1, 3))
    V_t = (V @ W_v).reshape((V.shape[0], V.shape[1], num_heads, dk)).transpose((0, 2, 1, 3))

    heads = softmax((Q_t @ K_t.transpose(0, 1, 3, 2)) / np.sqrt(dk)) @ V_t
    return heads.transpose((0, 2, 1, 3)).reshape((heads.shape[0], Q.shape[1], W_v.shape[0])) @ W_o

    