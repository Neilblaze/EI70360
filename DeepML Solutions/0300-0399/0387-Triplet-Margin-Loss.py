import numpy as np

def triplet_margin_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 1.0) -> float:
    """
    Compute the triplet margin loss for metric learning.
    
    Args:
        anchor: Anchor embeddings, shape (D,) for single or (N, D) for batch
        positive: Positive embeddings (same class as anchor), same shape as anchor
        negative: Negative embeddings (different class from anchor), same shape as anchor
        margin: Minimum desired distance gap between positive and negative pairs
    
    Returns:
        Mean triplet margin loss as a float
    """
    a = np.asarray(anchor, dtype=float)
    p = np.asarray(positive, dtype=float)
    n = np.asarray(negative, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
        p = p.reshape(1, -1)
        n = n.reshape(1, -1)

    d_ap = np.sqrt(np.sum((a - p) ** 2, axis=1))
    d_an = np.sqrt(np.sum((a - n) ** 2, axis=1))
    losses = np.maximum(0.0, d_ap - d_an + margin)
    return float(np.mean(losses))
