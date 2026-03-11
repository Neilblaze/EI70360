import numpy as np


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        h = initial_hidden_state
        c = initial_cell_state
        outputs = []

        for item in x:
            z = np.row_stack((h, item.reshape(-1, 1)))

            f = self.sigmoid(np.dot(self.Wf, z) + self.bf)
            i = self.sigmoid(np.dot(self.Wi, z) + self.bi)
            c_tilde = self.tanh(np.dot(self.Wc, z) + self.bc)

            c = f * c + i * c_tilde

            o = self.sigmoid(np.dot(self.Wo, z) + self.bo)
            h = o * self.tanh(c)

            outputs.append(h)

        return np.array(outputs), h, c
