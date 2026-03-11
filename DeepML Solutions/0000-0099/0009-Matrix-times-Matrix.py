def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]]:
    if len(a[0]) != len(b):
        return -1

    c = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]

    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]

    return c


A = [[1, 2], [2, 4]]

B = [[2, 1], [3, 4]]

c = matrixmul(A, B)
print(c)
