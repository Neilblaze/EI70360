def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
    n = len(input_matrix)
    m = len(input_matrix[0])

    result = []
    for i in range(0, n, pool_size):
        row = []
        for j in range(0, m, pool_size):
            total = 0
            count = 0
            for ki in range(i, i + pool_size):
                for kj in range(j, j + pool_size):
                    total += input_matrix[ki][kj]
                    count += 1

            row.append(total / count)

        result.append(row)

    return result
