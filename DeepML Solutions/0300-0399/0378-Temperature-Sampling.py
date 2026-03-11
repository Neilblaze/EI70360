import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float) -> list:
	"""
	Compute temperature-scaled softmax probabilities from logits.
	
	Args:
		logits: 1D numpy array of raw model output scores
		temperature: float controlling distribution sharpness
	
	Returns:
		List of probabilities after temperature scaling
	"""
    if temperature <= 0:
        probs = np.zeros_like(logits)
        max_index = np.argmax(logits)
        probs[max_index] = 1.0
        return probs.tolist()

	scaled = logits / temperature
	scaled = scaled - np.max(scaled)
	exp_vals = np.exp(scaled)
	probs = exp_vals / np.sum(exp_vals)
	return probs.tolist()
