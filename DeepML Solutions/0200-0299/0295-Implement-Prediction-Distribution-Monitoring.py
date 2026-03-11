import numpy as np

def monitor_prediction_distribution(reference_preds: list, current_preds: list, n_bins: int = 10) -> dict:
    """
    Monitor prediction distribution changes between reference and current predictions.
    
    Args:
        reference_preds: List of reference prediction scores (floats between 0 and 1)
        current_preds: List of current prediction scores (floats between 0 and 1)
        n_bins: Number of bins for histogram comparison
    
    Returns:
        Dictionary with keys: 'mean_shift', 'std_ratio', 'js_divergence', 'drift_detected'
    """
    ref = np.asarray(reference_preds, dtype=float)
    cur = np.asarray(current_preds, dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return None

    mean_shift = np.mean(cur) - np.mean(ref)

    ref_std = np.std(ref)
    cur_std = np.std(cur)

    if ref_std == 0:
        std_ratio = np.inf
    else:
        std_ratio = cur_std / ref_std

    bins = np.linspace(0, 1, n_bins + 1)

    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)

    ref_probs = (ref_counts + 1) / (len(ref) + n_bins)
    cur_probs = (cur_counts + 1) / (len(cur) + n_bins)

    M = 0.5 * (ref_probs + cur_probs)
    kl_divergence = lambda P, Q: np.sum(P * np.log(P / Q))
    js_divergence = 0.5 * kl_divergence(ref_probs, M) + 0.5 * kl_divergence(cur_probs, M)
    drift_detected = js_divergence > 0.1

    return {
        "mean_shift": round(float(mean_shift), 4),
        "std_ratio": round(float(std_ratio), 4),
        "js_divergence": round(float(js_divergence), 4),
        "drift_detected": drift_detected,
    }
