import numpy as np

def svm_margin_width(w: np.ndarray) -> float:
    """
    Calculate the margin width of a linear SVM classifier.
    
    Parameters:
    w : np.ndarray - weight vector defining the hyperplane
    
    Returns:
    float - the total margin width
    """
    sum_ = sum(np.power(w, 2))
    sqrt_ = np.sqrt(sum_)
    return 2 / sqrt_
