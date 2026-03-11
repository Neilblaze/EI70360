def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.
    
    Args:
        initial_lr: Starting learning rate
        end_lr: Final learning rate
        num_steps: Total number of training steps
    
    Returns:
        List of learning rates for each step
    """
    if num_steps == 0:
        return []

    result = [initial_lr]
    
    if num_steps == 1:
        return result

    decay = (initial_lr - end_lr) / (num_steps - 1)

    for _ in range(1, num_steps):
        result.append(result[-1] - decay)

    return result
