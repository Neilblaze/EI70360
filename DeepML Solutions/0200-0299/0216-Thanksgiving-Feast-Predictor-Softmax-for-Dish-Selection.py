import numpy as np


def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
    """
    Predict the probability of choosing each Thanksgiving dish using softmax.
    
    Args:
        preference_scores: List of preference scores for each dish
        (e.g., [turkey_score, stuffing_score, cranberry_score, pie_score])
        
    Returns:
        List of probabilities for each dish
    """
    e_x = np.exp(preference_scores - np.max(preference_scores))
    result = e_x / np.sum(e_x, axis=0)
    return [round(item, 4) for item in result]

result = thanksgiving_dish_predictor(
    preference_scores=[2.0, 1.0, 0.5, 1.5],
   
)
print(result)
