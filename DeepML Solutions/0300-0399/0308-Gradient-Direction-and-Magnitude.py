import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
    if any(x == 0 for x in gradient):
        return {
            "magnitude": 0.0,
            "direction": [0.0 for _ in gradient],
            "descent_direction": [0.0 for _ in gradient],
        }

	pow_gradient = [x ** 2 for x in gradient]
    sum_pow_gradient = sum(pow_gradient)
    magnitude = sum_pow_gradient ** 0.5
    direction = [x / magnitude for x in gradient]
    descent_direction = [-x for x in direction]
    return {
        "magnitude": round(magnitude, 4),
        "direction": direction,
        "descent_direction": descent_direction,
    }
