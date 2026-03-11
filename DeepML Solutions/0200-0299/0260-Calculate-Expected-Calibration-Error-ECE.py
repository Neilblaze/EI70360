import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_pred >= bin_lower, y_pred < bin_upper)
        if bin_upper == 1.0:
            in_bin = np.logical_or(in_bin, y_pred == 1.0)
        
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            mean_pred_in_bin = np.mean(y_pred[in_bin])
            mean_actual_in_bin = np.mean(y_true[in_bin])
            ece += np.abs(mean_pred_in_bin - mean_actual_in_bin) * prop_in_bin
    
    return round(ece, 3)
