def calculate_sla_metrics(requests: list, latency_sla_ms: float = 100.0) -> dict:
    """
    Calculate SLA compliance metrics for a model serving endpoint.
    
    Args:
        requests: list of request results, each a dict with 'latency_ms' and 'status'
        latency_sla_ms: maximum acceptable latency in ms for SLA compliance
    
    Returns:
        dict with keys: 'latency_sla_compliance', 'error_rate', 'overall_sla_compliance'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not requests:
        return {}
    
    total_requests = len(requests)
    successful_requests = 0
    successful_and_fast = 0
    failed_requests = 0

    for request in requests:
        status = request['status']
        latency = request['latency_ms']
        
        if status == 'success':
            successful_requests += 1
            if latency <= latency_sla_ms:
                successful_and_fast += 1
        else:
            failed_requests += 1

    latency_sla_compliance = (successful_and_fast / successful_requests * 100) if successful_requests > 0 else 0.0
    error_rate = (failed_requests / total_requests) * 100
    overall_sla_compliance = (successful_and_fast / total_requests) * 100

    return {
        'latency_sla_compliance': round(latency_sla_compliance, 2),
        'error_rate': round(error_rate, 2),
        'overall_sla_compliance': round(overall_sla_compliance, 2)
    }

requests = [
    {'status': 'success', 'latency_ms': 50}, 
    {'status': 'success', 'latency_ms': 80}, 
    {'status': 'success', 'latency_ms': 120}, 
    {'status': 'error', 'latency_ms': 30}, 
    {'status': 'timeout', 'latency_ms': 5000}
]
latency_sla_ms = 100.0
result = calculate_sla_metrics(requests, latency_sla_ms)
print(result)