import numpy as np


def batch_iterator(X, y=None, batch_size=64):
    n_batches = len(X) // batch_size
    X_batches = np.array_split(X[: n_batches * batch_size], n_batches)

    if y is not None:
        y_batches = np.array_split(y[: n_batches * batch_size], n_batches)

        if len(X) % batch_size != 0:
            X_batches.append(X[n_batches * batch_size :])
            y_batches.append(y[n_batches * batch_size :])

        return list(zip(X_batches, y_batches))
    else:
        if len(X) % batch_size != 0:
            X_batches.append(X[n_batches * batch_size :])
        return X_batches


X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

y = np.array([1, 2, 3, 4, 5])
batch_size = 2
batches = batch_iterator(X=X, y=y, batch_size=batch_size)
print(batches)
