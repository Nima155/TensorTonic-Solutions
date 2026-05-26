import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    """
    Returns: dict with "d_loss" and "g_loss" as float values
    """
    # Your implementation here
    real_data = np.array(real_data)
    fake_data = np.array(fake_data)
    D_W = np.array(D_W)
    p_f = (1 / (1 + np.exp(-fake_data @ D_W))).clip(1e-8, 1-1e-8)
    p_r = (1 / (1 + np.exp(-real_data @ D_W))).clip(1e-8, 1-1e-8)
    dl = -np.mean(np.log(p_r) + np.log(1 - p_f))
    gl = -np.mean(np.log(p_f))
    return {
        "d_loss": dl, "g_loss": gl
    }