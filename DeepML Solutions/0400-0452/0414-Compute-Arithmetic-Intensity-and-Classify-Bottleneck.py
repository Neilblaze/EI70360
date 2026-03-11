def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    arithmetic_intensity = flops / bytes_accessed
    ridge_point = peak_performance / peak_bandwidth
    if arithmetic_intensity < ridge_point:
        bottleneck = "memory-bound"
        achieved_performance = float(arithmetic_intensity * peak_bandwidth)
    else:
        bottleneck = "compute-bound"
        achieved_performance = float(peak_performance)

    utilization_percent = (achieved_performance / peak_performance) * 100
    return {
        "arithmetic_intensity": round(arithmetic_intensity, 4),
        "ridge_point": round(ridge_point, 4),
        "bottleneck": bottleneck,
        "achieved_performance": round(achieved_performance, 4),
        "utilization_percent": round(utilization_percent, 4),
    }
