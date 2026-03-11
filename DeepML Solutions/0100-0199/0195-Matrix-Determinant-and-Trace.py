import numpy as np

def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
	"""
	Compute the determinant and trace of a square matrix.
	
	Args:
		matrix: A square matrix (n x n) represented as list of lists

	Returns:
		Tuple of (determinant, trace)
	"""
	det = float(np.linalg.det(matrix))
    diag_sum = float(sum(matrix[i][i] for i in range(len(matrix))))
    return (det, diag_sum)

matrix = [[2, 3], [1, 4]]
result = matrix_determinant_and_trace(matrix)
print(result)
