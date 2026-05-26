import numpy as np

def discriminator_loss(real_probs, fake_probs):
    """Compute discriminator loss using binary cross-entropy.
    Returns: Loss value rounded to 4 decimals."""
    real_probs = np.array(real_probs).clip(1e-8, 1-1e-8)
    fake_probs = np.array(fake_probs).clip(1e-8, 1-1e-8) 
   
    return -(np.log(real_probs) + np.log((1 - fake_probs))).mean()

def generator_loss(fake_probs):
    """Compute non-saturating generator loss.
    Returns: Loss value rounded to 4 decimals."""
    fake_probs = np.array(fake_probs).clip(1e-8, 1-1e-8)
    return -np.log(fake_probs ).mean()