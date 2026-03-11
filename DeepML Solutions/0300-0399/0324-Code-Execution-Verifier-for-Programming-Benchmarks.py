import numpy as np

def verify_code_execution(
    test_cases: list[dict],
    numeric_tolerance: float = 1e-6
) -> dict:
    """
    Verify code execution results for a programming benchmark.
    
    Args:
        test_cases: List of dicts with keys:
            - 'expected': Expected output string
            - 'actual': Actual output string (or None if execution failed)
            - 'status': 'success', 'error', or 'timeout'
        numeric_tolerance: Tolerance for floating-point comparisons
        
    Returns:
        Dict with keys:
            - 'pass_rate': Proportion of passed tests (float, rounded to 4 decimals)
            - 'error_rate': Proportion of execution errors (float, rounded to 4 decimals)
            - 'passed_count': Number of passed tests (int)
            - 'total_count': Total number of tests (int)
            - 'verdicts': List of 'pass', 'fail', or 'error' for each test
    """
    if not test_cases:
        return {
            "pass_rate": 0.0,
            "error_rate": 0.0,
            "passed_count": 0,
            "total_count": 0,
            "verdicts": [],
        }

    verdicts = []
    passed_count = 0
    error_count = 0

    for test in test_cases:
        status = test["status"]

        if status != "success":
            verdicts.append("error")
            error_count += 1
            continue

        expected = test["expected"].strip()
        actual = test["actual"].strip() if test["actual"] is not None else ""

        try:
            exp_num = float(expected)
            act_num = float(actual)

            if abs(exp_num - act_num) < numeric_tolerance:
                verdicts.append("pass")
                passed_count += 1
            else:
                verdicts.append("fail")

        except ValueError:
            if expected == actual:
                verdicts.append("pass")
                passed_count += 1
            else:
                verdicts.append("fail")

    total_count = len(test_cases)
    pass_rate = passed_count / total_count if total_count > 0 else 0.0
    error_rate = error_count / total_count if total_count > 0 else 0.0

    return {
        "pass_rate": round(pass_rate, 4),
        "error_rate": round(error_rate, 4),
        "passed_count": passed_count,
        "total_count": total_count,
        "verdicts": verdicts,
    }
