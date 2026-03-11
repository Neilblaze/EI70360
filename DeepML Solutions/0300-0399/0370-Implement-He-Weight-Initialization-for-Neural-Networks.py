import numpy as np

def he_initialize(layer_dims: list, method: str = 'normal', seed: int = 42) -> list:
    """
    Initialize weight matrices for a neural network using He initialization.
    
    Args:
        layer_dims: List of integers representing neurons per layer
        method: 'normal' or 'uniform' sampling distribution
        seed: Random seed for reproducibility
    
    Returns:
        List of numpy arrays, one weight matrix per adjacent layer pair
    """
    np.random.seed(seed)
    weights = []
    n = len(layer_dims)
    for i in range(n - 1):
        fan_in = layer_dims[i]
        fan_out = layer_dims[i + 1]
        if method == "normal":
            std = np.sqrt(2.0 / fan_in)
            W = np.random.randn(fan_in, fan_out) * std

        elif method == "uniform":
            limit = np.sqrt(6.0 / fan_in)
            W = np.random.uniform(-limit, limit, (fan_in, fan_out))

        else:
            raise ValueError()

        weights.append(W)

    return weights
