def determinant_4x4(matrix: list[list[int | float]]) -> float:
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[0][2]
    d = matrix[0][3]
    e = matrix[1][0]
    f = matrix[1][1]
    g = matrix[1][2]
    h = matrix[1][3]
    i = matrix[2][0]
    j = matrix[2][1]
    k = matrix[2][2]
    l = matrix[2][3]
    m = matrix[3][0]
    n = matrix[3][1]
    o = matrix[3][2]
    p = matrix[3][3]

    det = (
        (
            a
            * (
                (f * ((k * p) - (l * o)))
                - (g * ((j * p) - (l * n)))
                + (h * ((j * o) - (k * n)))
            )
        )
        - (
            b
            * (
                (e * ((k * p) - (l * o)))
                - (g * ((i * p) - (l * m)))
                + (h * ((i * o) - (k * m)))
            )
        )
        + (
            c
            * (
                (e * ((j * p) - (l * n)))
                - (f * ((i * p) - (l * m)))
                + (h * ((i * n) - (j * m)))
            )
        )
        - (
            d
            * (
                (e * ((j * o) - (k * n)))
                - (f * ((i * o) - (k * m)))
                + (g * ((i * n) - (j * m)))
            )
        )
    )

    return det


matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
result = determinant_4x4(matrix)
print(result)
