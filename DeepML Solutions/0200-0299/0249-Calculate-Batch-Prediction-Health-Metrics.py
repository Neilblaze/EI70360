def calculate_batch_health(predictions: list, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        predictions: list of prediction results, each a dict with 'status' and optionally 'confidence'
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not predictions:
        return {}

    n = len(predictions)
    success_count = 0
    total_confidence = 0.0
    low_confidence_count = 0
    for prediction in predictions:
        if prediction['status'] == 'success':
            success_count += 1
            confidence = prediction['confidence']
            total_confidence += confidence
            if confidence < confidence_threshold:
                low_confidence_count += 1

    if success_count > 0:
        success_rate = round(100 * success_count / n, 2)
        avg_confidence = round(100 * total_confidence / success_count, 2)
        low_confidence_rate = round(100 * low_confidence_count / success_count, 2)
    else:
        success_rate = 0.0
        avg_confidence = 0.0
        low_confidence_rate = 0.0
    

    return {
        'success_rate': success_rate,
        'avg_confidence': avg_confidence,
        'low_confidence_rate': low_confidence_rate,
    }

predictions = [
    {'status': 'success', 'confidence': 0.9}, 
    {'status': 'success', 'confidence': 0.8}, 
    {'status': 'error'}, 
    {'status': 'success', 'confidence': 0.4}, 
    {'status': 'success', 'confidence': 0.7}
]
confidence_threshold = 0.5
result = calculate_batch_health(predictions, confidence_threshold)
print(result)