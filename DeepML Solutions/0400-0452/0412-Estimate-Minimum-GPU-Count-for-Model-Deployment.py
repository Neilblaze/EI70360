import math

def estimate_min_gpus(num_params_billion: float, bytes_per_param: int, gpu_memory_gb: float, overhead_fraction: float) -> dict:
    """
    Estimate the minimum number of GPUs needed to deploy a model.
    
    Args:
        num_params_billion: Number of model parameters in billions
        bytes_per_param: Bytes per parameter (4=FP32, 2=FP16, 1=INT8)
        gpu_memory_gb: Available memory per GPU in GB
        overhead_fraction: Fraction of model memory for runtime overhead
    
    Returns:
        dict with 'model_memory_gb', 'total_memory_gb', 'min_gpus'
    """
    model_memory_gb = float(num_params_billion * bytes_per_param)
    total_memory_gb = model_memory_gb * (1 + overhead_fraction)
    min_gpu = math.ceil(total_memory_gb / gpu_memory_gb)
    return {
        "model_memory_gb": round(model_memory_gb, 2),
        "total_memory_gb": round(total_memory_gb, 2),
        "min_gpus": min_gpu,
    }
