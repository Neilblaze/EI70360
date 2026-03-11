import math
from collections import Counter


def disorder(apples: list) -> float:
    color_counts = Counter(apples)
    total_apples = len(apples)

    entropy = 0.0
    for count in color_counts.values():
        probability = count / total_apples
        entropy -= probability * math.log2(probability)

    return entropy
