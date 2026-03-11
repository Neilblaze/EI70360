def min_max(x: list[int]) -> list[float]:
    result = []

    xmin = min(x)
    xmax = max(x)

    for item in x:
        try:
            result.append((item - xmin) / (xmax - xmin))

        except ZeroDivisionError:
            result.append(0.0)

    return result


print(min_max([1, 2, 3, 4, 5]))
print(min_max([5, 5, 5, 5, 5]))
