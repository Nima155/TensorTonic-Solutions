import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    """
    Returns: dict with "diversity_score" (float) and "is_collapsed" (bool)
    """
    # Your implementation here
    ds = np.sqrt(np.array(generated_samples).var(axis=0)).mean()
    return {"diversity_score": ds, "is_collapsed": ds < threshold}