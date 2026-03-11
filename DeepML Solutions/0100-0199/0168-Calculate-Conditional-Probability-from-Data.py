def conditional_probability(data, x, y):
    p_0 = 0
    p_1 = 0
    for item in data:
        if item[0] == x and item[1] == y:
            p_0 += 1
        elif item[0] == x and item[1] != y:
            p_1 += 1

    if not p_0 + p_1:
        return 0.0

    result = p_0 / (p_0 + p_1)
    return round(result, 4)
