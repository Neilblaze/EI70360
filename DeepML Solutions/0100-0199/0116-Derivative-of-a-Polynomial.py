def poly_term_derivative(c: float = 2.0, x: float = 3.0, n: float = 2.0) -> float:
    if n == 0:
        return 0.0
    else:
        return float(c * n * (x ** (n - 1)))

print(poly_term_derivative(2, 3, 2))