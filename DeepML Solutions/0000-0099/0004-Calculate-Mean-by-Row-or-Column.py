def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "column":
        means = [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == "row":
        means = [sum(row) / len(row) for row in matrix]

    return means


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mode = "column"
means = calculate_matrix_mean(matrix, mode)
print(means)
