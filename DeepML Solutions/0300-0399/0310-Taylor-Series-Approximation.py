import numpy as np
import math

def taylor_approximation(func_name: str, x: float, n_terms: int) -> float:
    result = 0.0

    if func_name == "exp":
        for n in range(n_terms):
            result += (x ** n) / math.factorial(n)

    elif func_name == "sin":
        for n in range(n_terms):
            term_n = 2 * n + 1
            sign = 1 if n % 2 == 0 else -1
            result += sign * (x ** term_n) / math.factorial(term_n)

    elif func_name == "cos":
        for n in range(n_terms):
            term_n = 2 * n
            sign = 1 if n % 2 == 0 else -1
            result += sign * (x ** term_n) / math.factorial(term_n)

    return round(result, 6)
