import numpy as np

def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0) -> float:
	"""
	Compute the polynomial kernel between two vectors.
	
	Args:
		x: First input vector
		y: Second input vector
		degree: Degree of the polynomial (default: 3)
		gamma: Scaling factor (default: 1.0)
		coef0: Independent term in kernel function (default: 1.0)
	
	Returns:
		The polynomial kernel value as a float
	"""
    x = np.array(x)
    y = np.array(y)
    dot_product = np.dot(x, y)
    return (gamma * dot_product + coef0) ** degree
