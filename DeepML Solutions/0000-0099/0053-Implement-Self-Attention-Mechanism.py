import numpy as np


def self_attention(Q, K, V):
    Q_dim = np.array([float(Q.shape[0]) ** 0.5], dtype="float32")
    attention_score = np.divide(Q.dot(K.transpose()), Q_dim)
    softmax_attention_score = np.exp(attention_score) / np.sum(
        np.exp(attention_score), axis=-1, keepdims=True
    )
    return softmax_attention_score.dot(V)


def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


X = np.array([[1, 1], [1, 0]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)
