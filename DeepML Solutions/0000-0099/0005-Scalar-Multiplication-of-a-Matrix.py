def scalar_multiply(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    result = []
    for row in matrix:
        result.append([scalar * item for item in row])

    return result


matrix = [[1, 2], [3, 4]]
scalar = 2
result = scalar_multiply(matrix, scalar)
print(result)
