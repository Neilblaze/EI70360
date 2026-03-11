import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)


# DO NOT CHANGE LAYER CLASS
class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        limit = 1 / math.sqrt(input_shape[0])
        self.W = np.random.uniform(-limit, limit, size=(input_shape[0], n_units))
        self.w0 = 0

    def initialize(self, optimizer):
        self.W_optimizer = copy.copy(optimizer)
        self.b_optimizer = copy.copy(optimizer)

    def forward_pass(self, X):
        self.layer_input = X
        return np.dot(X, self.W) + self.w0

    def backward_pass(self, accum_grad):
        dW = self.layer_input.T.dot(accum_grad)
        db = np.sum(accum_grad, axis=0, keepdims=True)
        accum_grad = accum_grad.dot(self.W.T)

        self.W = self.W_optimizer.update(self.W, dW)
        self.w0 = self.b_optimizer.update(self.w0, db)

        return accum_grad

    def number_of_parameters():
        pass


# Initialize a Dense layer with 3 neurons and input shape (2,)
dense_layer = Dense(n_units=3, input_shape=(2,))


# Define a mock optimizer with a simple update rule
class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad


optimizer = MockOptimizer()

# Initialize the Dense layer with the mock optimizer
dense_layer.initialize(optimizer)

# Perform a forward pass with sample input data
X = np.array([[1, 2]])
output = dense_layer.forward_pass(X)
print("Forward pass output:", output)

# Perform a backward pass with sample gradient
accum_grad = np.array([[0.1, 0.2, 0.3]])
back_output = dense_layer.backward_pass(accum_grad)
print("Backward pass output:", back_output)
