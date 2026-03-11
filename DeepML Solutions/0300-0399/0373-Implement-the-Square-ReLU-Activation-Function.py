import numpy as np

def square_relu(x: np.ndarray) -> dict:
	"""
	Apply the Square ReLU activation function and compute its derivative.
	
	Args:
		x: Input numpy array of any shape
	
	Returns:
		Dictionary with 'output' and 'derivative' as numpy arrays
	"""
	output = np.where(x > 0, x ** 2, 0.0)
	output = np.round(output, 4)

	derivative = np.where(x > 0, 2 * x, 0.0)
	derivative = np.round(derivative, 4)
	return { "output": output, "derivative": derivative }
