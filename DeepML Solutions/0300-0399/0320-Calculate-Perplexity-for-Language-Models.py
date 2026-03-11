import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    log_probs = np.log(probabilities)
    avg_log_prob = -np.mean(log_probs)
    perplexity = np.exp(avg_log_prob)
    return perplexity
