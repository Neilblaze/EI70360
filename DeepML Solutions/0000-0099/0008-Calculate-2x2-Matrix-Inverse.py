def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]
    inverse_func = 1 / ((a * d) - (b * c))
    inverse = [
        [inverse_func * d, inverse_func * -b],
        [inverse_func * -c, inverse_func * a],
    ]
    return inverse


matrix = [[4, 7], [2, 6]]
inverse = inverse_2x2(matrix)
print(inverse)
