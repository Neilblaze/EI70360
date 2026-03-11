import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    thresholds = np.unique(y_scores_sorted)[::-1]
    
    precisions = []
    recalls = []
    
    actual_positives = np.sum(y_true == 1)
    
    for threshold in thresholds:
        predicted_positives = np.sum(y_scores >= threshold)
        true_positives = np.sum((y_scores >= threshold) & (y_true == 1))
        
        if predicted_positives == 0:
            precision = 1.0
        else:
            precision = true_positives / predicted_positives
        
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = true_positives / actual_positives
        
        precisions.append(precision)
        recalls.append(recall)
    
    return (precisions, recalls, thresholds.tolist())
