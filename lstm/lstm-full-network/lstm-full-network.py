import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.c_last = np.zeros(hidden_dim)
        self.h_last = np.zeros(hidden_dim)
        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)

        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        Forward pass. Returns (y, h_last, C_last).
        """
        h_prev = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        # return [h_prev.shape, X.shape, self.hidden_dim]
        ys = []
        hs = []
        zs = []
        for x in X:
            h_prev = np.zeros(self.hidden_dim)
            c = np.zeros(self.hidden_dim)
            y_mid = []
            for r in x:
                f = sigmoid((np.concatenate([h_prev, r], axis=-1)) @ self.W_f.T + self.b_f)
                i_t = sigmoid((np.concatenate([h_prev, r], axis=-1)) @ self.W_i.T + self.b_i)
                g = np.tanh((np.concatenate([h_prev, r], axis=-1)) @ self.W_c.T + self.b_c)
                o_t = sigmoid((np.concatenate([h_prev, r], axis=-1)) @ self.W_o.T + self.b_o)
                c = f * c + i_t * g
                h_prev = o_t * np.tanh(c)
                y_mid.append((self.W_y @ h_prev) + self.b_y)  
            hs.append(h_prev)      
            zs.append(c)
            ys.append(y_mid)
            
        # return [f.shape, self.c_last.shape]
        return (ys, hs, zs)
        
        