import math

def tanh(x: float) -> float:
	"""
	Implements the Tanh (hyperbolic tangent) activation function.

	Args:
		x (float): Input value

	Returns:
		float: The tanh of the input, rounded to 4 decimal places
	"""
	result = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    return round(result, 4)
