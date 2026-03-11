def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    if not latencies_ms:
        return {}

    n = len(latencies_ms)
    sorted_latencies = sorted(latencies_ms)
    
    average_latency = sum(latencies_ms) / n
    throughput_per_sec = 1000 / average_latency if average_latency > 0 else 0

    def get_percentile(p):
        index = (n - 1) * p / 100
        floor_idx = int(index)
        frac = index - floor_idx
        if floor_idx == n - 1:
            return sorted_latencies[floor_idx]
        
        return sorted_latencies[floor_idx] + frac * (sorted_latencies[floor_idx + 1] - sorted_latencies[floor_idx])
    
    p50 = get_percentile(50)
    p95 = get_percentile(95)
    p99 = get_percentile(99)

    return {
        'throughput_per_sec': round(throughput_per_sec, 2),
        'avg_latency_ms': round(average_latency, 2),
        'p50_ms': round(p50, 2),
        'p95_ms': round(p95, 2),
        'p99_ms': round(p99, 2),    
    }


latencies_ms = [10, 20, 30, 40, 50]
# {'throughput_per_sec': 33.33, 'avg_latency_ms': 30.0, 'p50_ms': 30.0, 'p95_ms': 48.0, 'p99_ms': 49.6}
result = calculate_inference_stats(latencies_ms)
print(result)