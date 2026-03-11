import numpy as np


def calculate_contrast(img) -> int:
    return round(np.max(img) - np.min(img), 3)
