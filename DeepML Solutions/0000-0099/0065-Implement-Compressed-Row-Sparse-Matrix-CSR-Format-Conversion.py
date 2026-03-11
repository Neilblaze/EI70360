import numpy as np


def compressed_row_sparse_matrix(dense_matrix):
    vals = []
    col_idx = []
    row_ptr = [0]

    for row in dense_matrix:
        for i in range(len(row)):
            if row[i] != 0:
                vals.append(row[i])
                col_idx.append(i)

        row_ptr.append(len(vals))

    return vals, col_idx, row_ptr


dense_matrix = [[1, 0, 0, 0], [0, 2, 0, 0], [3, 0, 4, 0], [1, 0, 0, 5]]

vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
print("Values array:", vals)
print("Column indices array:", col_idx)
print("Row pointer array:", row_ptr)
