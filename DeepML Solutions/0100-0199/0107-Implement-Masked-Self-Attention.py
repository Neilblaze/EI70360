import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def masked_attention(Q, K, V, mask=None):
    Q_dim = np.array([float(Q.shape[0]) ** 0.5], dtype="float32")

    attention_score = np.divide(Q.dot(K.transpose()), Q_dim)

    if mask is not None:
        attention_score += mask

    exp_score = np.exp(
        attention_score - np.max(attention_score, axis=-1, keepdims=True)
    )
    softmax_attention_score = exp_score / np.sum(exp_score, axis=-1, keepdims=True)
    return softmax_attention_score.dot(V)
