import math


def single_neuron_model(
    features: list[list[float]], labels: list[int], weights: list[float], bias: float
) -> (list[float], float):
    def sigmoid(z: float) -> float:
        result = 1 / (1 + math.exp(-z))
        return result

    def mse_loss(probs: list[float], labels: list[int]) -> float:
        result = 0
        for prob, label in zip(probs, labels):
            result += pow(prob - label, 2) / len(labels)

        result = round(result, 4)
        return result

    probs = []
    for feature in features:
        z = sum(f * weight for weight, f in zip(weights, feature)) + bias
        prob = sigmoid(z)
        prob = round(prob, 4)
        probs.append(prob)

    mse = mse_loss(probs, labels)

    return probs, mse


features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
labels = [0, 1, 0]
weights = [0.7, -0.4]
bias = -0.1

probabilities, mse = single_neuron_model(features, labels, weights, bias)
print(probabilities, mse)
