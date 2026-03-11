def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
    """
    Determine at each epoch whether training should stop based on validation loss.
    
    Args:
        val_losses: List of validation losses at each epoch
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as improvement
    
    Returns:
        List of booleans indicating whether to stop at each epoch
    """
    if not val_losses:
        return []

    best_loss = val_losses[0]
    counter = 0
    result = []

    for i, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            counter = 0
            result.append(False)
        else:
            if i == 0:
                result.append(False)
                continue
        
            counter += 1
            if counter >= patience:
                result.append(True)
            else:
                result.append(False)

    return result

result = early_stopping(
    val_losses=[0.5, 0.4, 0.39, 0.39, 0.39],
    patience=3,
    min_delta=0.01
)
print(result)
