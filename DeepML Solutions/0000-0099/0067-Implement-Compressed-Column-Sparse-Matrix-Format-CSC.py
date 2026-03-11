import numpy as np


def compressed_col_sparse_matrix(dense_matrix):
    vals = []
    col_idx = []
    col_ptr = [0]

    n_col = len(dense_matrix[0])
    n_row = len(dense_matrix)

    for col in range(n_col):
        for row in range(n_row):
            if dense_matrix[row][col] != 0:
                vals.append(dense_matrix[row][col])
                col_idx.append(row)

        col_ptr.append(len(vals))

    return vals, col_idx, col_ptr


dense_matrix = [[0, 0, 3, 0], [1, 0, 0, 4], [0, 2, 0, 0]]

vals, col_idx, row_ptr = compressed_col_sparse_matrix(dense_matrix)
print("Values array:", vals)
print("Column indices array:", col_idx)
print("Row pointer array:", row_ptr)
