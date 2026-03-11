import string
import re

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    if not predictions or not references:
        return 0.0
        
    n = len(predictions)
    m = 0
    for a, b in zip(predictions, references):
        a = re.sub(r'[^\w\s]', '', a)
        a = re.sub(' +', ' ', a)
        a = a.lower()
        a = a.strip()

        b = re.sub(r'[^\w\s]', '', b)
        b = re.sub(' +', ' ', b)
        b = b.lower()
        b = b.strip()

        if a == b:
            m += 1

    return m / n
