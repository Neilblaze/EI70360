def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
    n, m = len(a), len(b)
    if n != m:
        return -1
	
    result = []
    for i in range(n): 
        result.append(a[i] + b[i])

    return result

a = [1, 3]
b = [4, 5]
result = vector_sum(a, b)
print(result)