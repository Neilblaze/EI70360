import numpy as np

def clip_gradients_by_value(gradients: np.ndarray, clip_value: float) -> np.ndarray:
    """
    Clip gradient values to be within [-clip_value, clip_value].
    
    Args:
        gradients: A numpy array representing gradients (any shape)
        clip_value: The maximum absolute value for any gradient element (non-negative)
    
    Returns:
        Clipped gradients with same shape as input
    """
    return np.clip(gradients, -clip_value, clip_value)
