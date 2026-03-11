from typing import Tuple


def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    n = len(val_losses)
    best_loss = val_losses[0]
    best_idx = 0
    current_patience = 0
    
    for i in range(1, n):
        current_loss = val_losses[i]
        
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_idx = i
            current_patience = 0
        else:
            current_patience += 1
            
        if current_patience >= patience:
            return (i, best_idx)
    
    return (n - 1, best_idx)