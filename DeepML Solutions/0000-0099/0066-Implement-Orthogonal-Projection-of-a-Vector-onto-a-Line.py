def orthogonal_projection(v, L):
    v_mul_L = sum([a * b for a, b in zip(v, L)])
    L_mul_L = sum([item * item for item in L])
    return [round((v_mul_L / L_mul_L) * item, 3) for item in L]


v = [3, 4]
L = [1, 0]

print(orthogonal_projection(v, L))
