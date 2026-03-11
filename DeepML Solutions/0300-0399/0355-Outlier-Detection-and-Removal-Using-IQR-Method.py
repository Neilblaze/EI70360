import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
	"""
	Detect and remove outliers using the IQR method.
	
	Args:
		data: List of numerical values
		k: IQR multiplier for determining outlier bounds (default 1.5)
	
	Returns:
		Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
	"""
	q1 = np.percentile(data, 25)
	q3 = np.percentile(data, 75)
	iqr = q3 - q1
	lower_bound = round(q1 - iqr * k, 4)
	upper_bound = round(q3 + iqr * k, 4)
	cleaned_data = []
	outlier_indices = []
	for i, x in enumerate(data):
		if x > upper_bound or x < lower_bound:
			outlier_indices.append(i)
		else:
			cleaned_data.append(x)
	return { "cleaned_data": cleaned_data, "outlier_indices": outlier_indices, "lower_bound": lower_bound, "upper_bound": upper_bound }
