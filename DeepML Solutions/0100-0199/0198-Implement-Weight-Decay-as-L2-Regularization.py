def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	updated = []
	for i in range(len(parameters)):
		group = []
		for j in range(len(parameters[i])):
			if apply_to_all[i]:
				update = lr * (gradients[i][j] + weight_decay * parameters[i][j])
			else:
				update = lr * gradients[i][j]

			group.append(parameters[i][j] - update)

		updated.append(group)

	return updated

result = apply_weight_decay(
	parameters=[[1.0, 2.0]],
	gradients=[[0.1, 0.2]],
	lr=0.1,
	weight_decay=0.01,
	apply_to_all=[True]
)
print(result)
