import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    if len(set(y_true)) == 1:
        return 0.0

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
        
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    tpr_list = np.array(tpr_list)
    fpr_list = np.array(fpr_list)
    auc = np.trapz(tpr_list, fpr_list)
    auc = np.abs(auc) if auc != 0.0 else auc
    return auc
