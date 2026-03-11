def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    if len(data) == 0 or degree < 0:
        return []

    arr = []
    for item in data:
        temp = []
        for i in range(degree + 1):
            temp.append(item**i)

        arr.append(temp)

    return arr


data = [1.0, 2.0]
degree = 2
print(phi_transform(data, degree))
