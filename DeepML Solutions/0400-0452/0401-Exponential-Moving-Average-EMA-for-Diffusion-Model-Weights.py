import numpy as np

def ema_update(ema_params, model_params_list, decay):
    """
    Compute the Exponential Moving Average of model parameters over training steps.
    
    Args:
        ema_params: numpy array, initial EMA parameters
        model_params_list: list of numpy arrays, model params at each training step
        decay: float, EMA decay rate in [0, 1]
    Returns:
        Final EMA parameters as a (nested) list, rounded to 4 decimal places
    """
    ema = np.asarray(ema_params, dtype=float)

    for params in model_params_list:
        params = np.asarray(params, dtype=float)
        ema = decay * ema + (1 - decay) * params

    return np.round(ema, 4).tolist()
