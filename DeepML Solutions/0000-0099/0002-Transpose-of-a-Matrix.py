def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    b = []
    row_len = len(a[0])
    col_len = len(a)

    for i in range(row_len):
        c = []
        for j in range(col_len):
            c.append(a[j][i])

        b.append(c)
    return b


a = [[1, 2, 3], [4, 5, 6]]
b = transpose_matrix(a)
print(b)
