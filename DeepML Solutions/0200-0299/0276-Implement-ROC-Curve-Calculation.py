import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    thresholds = np.sort(np.unique(y_scores))

    tpr_list = []
    fpr_list = []

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        TPR = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        FPR = float(FP / (FP + TN)) if (FP + TN) > 0 else 0.0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    tpr_list = [0.0] + tpr_list[::-1]
    fpr_list = [0.0] + fpr_list[::-1]
    return (fpr_list, tpr_list)
