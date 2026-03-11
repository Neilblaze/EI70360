import math

def mish(x: float) -> float:
	"""
	Compute the Mish activation function.

	Args:
		x (float): Input value

	Returns:
		float: Mish activation value rounded to 4 decimal places
	"""
	result = x * math.tanh(math.log1p(math.exp(x)))
	return round(result, 4)
