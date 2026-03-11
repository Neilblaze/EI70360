import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    if seed is not None:
        np.random.seed(seed)

    if mode == 'fan_in':
        n = n_in
    elif mode == 'fan_out':
        n = n_out

    std = np.sqrt(2.0 / n)

    if distribution == 'normal':
        weights = np.random.normal(0, std, (n_in, n_out))

    elif distribution == 'uniform':
        bound = np.sqrt(6.0 / n)
        weights = np.random.uniform(-bound, bound, (n_in, n_out))

    return weights
