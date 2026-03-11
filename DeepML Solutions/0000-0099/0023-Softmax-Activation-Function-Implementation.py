import math


def softmax(scores: list[float]) -> list[float]:
    probabilities = []

    for score in scores:
        prob = math.exp(score) / sum([math.exp(_) for _ in scores])
        prob = round(prob, 4)
        probabilities.append(prob)
    return probabilities


scores = [1, 2, 3]
probabilities = softmax(scores)
print(probabilities)
