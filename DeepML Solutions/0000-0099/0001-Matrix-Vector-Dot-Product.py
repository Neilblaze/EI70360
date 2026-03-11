def matrix_dot_vector(
    a: list[list[int | float]], b: list[int | float]
) -> list[int | float]:
    a_rows = len(a)
    a_cols = len(a[0])

    if a_cols != len(b):
        return -1

    result = []

    for row in a:
        r = 0
        for i in range(a_rows):
            r += row[i] * b[i]

        result.append(r)

    return result


result = matrix_dot_vector([[1, 2], [2, 4]], [1, 2])
print(result)
