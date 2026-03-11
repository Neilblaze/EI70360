def hard_voting_classifier(predictions: list[list[int]]) -> list[int]:
    """
    Implement a hard voting classifier using majority vote.
    
    Args:
        predictions: 2D list where predictions[i][j] is classifier i's prediction for sample j
        
    Returns:
        List of final predictions using majority vote
    """
    if not predictions:
        return []

    n_classifiers = len(predictions)
    n_samples = len(predictions[0])

    final_predictions = []

    for sample_idx in range(n_samples):
        votes = {}
        for classifier_idx in range(n_classifiers):
            vote = predictions[classifier_idx][sample_idx]
            votes[vote] = votes.get(vote, 0) + 1

        max_votes = max(votes.values())
        candidates = [cls for cls, count in votes.items() if count == max_votes]
        final_predictions.append(min(candidates))

    return final_predictions
