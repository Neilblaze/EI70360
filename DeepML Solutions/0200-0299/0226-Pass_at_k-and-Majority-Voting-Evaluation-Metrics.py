import numpy as np
from collections import Counter

def pass_at_1(responses_correct: np.ndarray) -> float:
    """
    Compute pass@1 by averaging correctness.

    Args:
        responses_correct: Boolean array for each response
        
    Returns:
        pass@1 score
    """
    if len(responses_correct) == 0:
        return 0.0
    
    return np.mean(responses_correct)


def majority_voting(responses: list[str]) -> str:
    """
    Return the most common response.

    Args:
        responses: List of response strings
        
    Returns:
        Most frequent response
    """
    if len(responses) == 0:
        return ""
    
    votes = Counter(responses)
    most_common = votes.most_common(1)[0][0]
    return most_common

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute unbiased pass@k from n samples with c correct.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total samples
        c: Correct samples
        k: k in pass@k
        
    Returns:
        Estimated pass@k
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    
    from math import comb
    pass_at_k_value = 1.0 - comb(n - c, k) / comb(n, k)
    return pass_at_k_value