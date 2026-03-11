import math


def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    
    Args:
        x: Input value

    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    sigmoid_val = 1 / (1 + math.exp(-x))
    sigmoid_derivative = sigmoid_val * (1 - sigmoid_val)
    tanh_val = math.tanh(x)
    tanh_derivative = 1 - tanh_val ** 2
    relu_derivative = 1 if x > 0 else 0
    return {
        'sigmoid': sigmoid_derivative,
        'tanh': tanh_derivative,
        'relu': relu_derivative
    }

result = activation_derivatives(x=0.0)
print(result)
