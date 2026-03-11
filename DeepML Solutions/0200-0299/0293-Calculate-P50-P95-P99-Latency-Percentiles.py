import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    if not latencies:
        return {'P50': 0.0, 'P95': 0.0, 'P99': 0.0}
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    def get_percentile(p):
        index = (n - 1) * p / 100.0
        floor_idx = int(index)
        frac = index - floor_idx
        if floor_idx >= n - 1:
            return sorted_latencies[-1]
        return sorted_latencies[floor_idx] + frac * (sorted_latencies[floor_idx + 1] - sorted_latencies[floor_idx])
    
    p50 = get_percentile(50)
    p95 = get_percentile(95)
    p99 = get_percentile(99)
    
    return {
        'P50': round(p50, 4),
        'P95': round(p95, 4),
        'P99': round(p99, 4)
    }
