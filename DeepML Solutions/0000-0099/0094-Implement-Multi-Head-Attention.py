import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)


def self_attention(Q, K, V):
    scores = np.matmul(Q, K.T) / np.sqrt(Q.shape[-1])
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights /= attention_weights.sum(axis=-1, keepdims=True)
    return np.matmul(attention_weights, V)


def multi_head_attention(Q, K, V, n_heads):
    d_k = Q.shape[1] // n_heads
    Q_reshaped = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)
    K_reshaped = K.reshape(K.shape[0], n_heads, d_k).transpose(1, 0, 2)
    V_reshaped = V.reshape(V.shape[0], n_heads, d_k).transpose(1, 0, 2)

    attentions = [
        self_attention(Q_reshaped[i], K_reshaped[i], V_reshaped[i])
        for i in range(n_heads)
    ]
    return np.concatenate(attentions, axis=-1)


Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 0], [0, 1]])
V = np.array([[1, 0], [0, 1]])
n_heads = 2

result = multi_head_attention(Q, K, V, n_heads)
print(result)
