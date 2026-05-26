import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        """
        Generate fake samples from noise z using tanh(z @ G_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        return np.tanh(np.array(z) @ self.G_W)
    
    def discriminate(self, x):
        """
        Classify samples using sigmoid(x @ D_W).
        Returns list of lists, rounded to 4 decimals.
        """
        # Your implementation here
        return 1 / (1 + np.exp(-np.array(x) @ self.D_W))
    
    def train_step(self, real_data, z):
        """
        Compute d_loss and g_loss for one training step.
        Returns dict with "d_loss" and "g_loss", rounded to 4 decimals.
        """
        # Your implementation here
        real_data = np.array(real_data)
        z = np.array(self.generate(z))
        p_f = (1 / (1 + np.exp(-z @ self.D_W))).clip(1e-8, 1-1e-8)
        p_r = (1 / (1 + np.exp(-real_data @ self.D_W))).clip(1e-8, 1-1e-8)
        dl = -np.mean(np.log(p_r) + np.log(1 - p_f))
        gl = -np.mean(np.log(p_f))
        return {
            "d_loss": dl, "g_loss": gl
        }