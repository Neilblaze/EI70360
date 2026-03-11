import numpy as np

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    np.random.seed(seed)
    shape = [fan_in, fan_out]

    if mode == "uniform":
        limit = np.sqrt(6 / (fan_in + fan_out))
        weights = np.random.uniform(-limit, limit, shape)
    elif mode == "normal":
        stddev = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.normal(0, stddev, shape)
    else:
        raise ValueError("""mode must be 'uniform' or 'normal'""")

    result = {
        "weights": np.round(weights, 4),
        "shape": shape,
        "param": np.round(limit, 4) if mode == "uniform" else np.round(stddev, 4)
    }
    return resulto
