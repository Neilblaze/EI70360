import numpy as np


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> (np.ndarray, float, list[float]):
    features, labels, weights = (
        np.array(features),
        np.array(labels),
        np.array(initial_weights),
    )
    mse_values = []
    bias = initial_bias

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = 1 / (1 + np.exp(-z))

        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        errors = predictions - labels
        weight_gradients = np.dot(features.T, errors * predictions * (1 - predictions))
        bias_gradient = np.sum(errors * predictions * (1 - predictions))

        weights -= learning_rate * weight_gradients / len(labels)
        bias -= learning_rate * bias_gradient / len(labels)

    return np.round(weights, 4).tolist(), round(bias, 4), mse_values


features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]
labels = [1, 0, 0]
initial_weights = [0.1, -0.2]
initial_bias = 0.0
learning_rate = 0.1
epochs = 2

updated_weights, updated_bias, mse_losses = train_neuron(
    features, labels, initial_weights, initial_bias, learning_rate, epochs
)
print(updated_weights, updated_bias, mse_losses)
