import numpy as np


def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1

    pos = np.arange(position, dtype=np.float32).reshape(position, 1)
    ind = np.arange(d_model, dtype=np.float32).reshape(1, d_model)
    angles = pos * (1 / np.power(10000, (2 * (ind // 2)) / np.float32(d_model)))
    sine = np.sin(angles[:, ::2])
    cosine = np.cos(angles[:, 1::2])
    pos_encoding = np.concatenate([sine, cosine], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, :]
    return np.float16(pos_encoding)


print(pos_encoding(2, 8))
