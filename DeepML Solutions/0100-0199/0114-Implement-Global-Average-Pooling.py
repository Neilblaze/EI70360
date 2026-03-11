import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    result = np.mean(x, axis=(1, 2))
    
    if squeeze_output:
        result = np.squeeze(result, axis=0)
    
    return result

x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
result = global_avg_pool(x)
print(result)