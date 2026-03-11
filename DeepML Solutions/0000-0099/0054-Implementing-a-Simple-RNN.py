import numpy as np


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    h = initial_hidden_state
    for x in input_sequence:
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
    return h


input_sequence = [[1.0], [2.0], [3.0]]
initial_hidden_state = [0.0]
Wx = [[0.5]]
Wh = [[0.8]]
b = [0.0]

final_hidden_state = rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b)
print(final_hidden_state)
