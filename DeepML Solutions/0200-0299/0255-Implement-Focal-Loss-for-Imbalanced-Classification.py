import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
	"""
	Compute Focal Loss for multi-class classification.
	
	Args:
		y_true: Ground truth labels as class indices (list or 1D array)
		y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
		gamma: Focusing parameter (default: 2.0)
		alpha: Class weights (optional, list or 1D array of length n_classes)
	
	Returns:
		float: Average focal loss
	"""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=float)
    
    n_samples = y_true.shape[0]
    n_classes = y_pred.shape[1]
    
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    pt = y_pred[np.arange(n_samples), y_true]
    
    ce = -np.log(pt)
    
    focal_weight = (1 - pt) ** gamma
    
    loss = focal_weight * ce
    
    if alpha is not None:
        alpha = np.array(alpha, dtype=float)
        if alpha.shape[0] == n_classes:
            class_weights = alpha[y_true]
            loss = class_weights * loss
    
    return float(np.mean(loss))
