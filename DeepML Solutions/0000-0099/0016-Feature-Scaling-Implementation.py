import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    standardized_data = (data - data_mean) / data_std

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    normalized_data = (data - data_min) / (data_max - data_min)

    return standardized_data, normalized_data


data = np.array([[1, 2], [3, 4], [5, 6]])
standardized_data, normalized_data = feature_scaling(data)
print(standardized_data, normalized_data)
