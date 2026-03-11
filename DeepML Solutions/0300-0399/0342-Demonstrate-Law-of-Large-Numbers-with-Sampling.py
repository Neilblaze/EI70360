import numpy as np

def law_of_large_numbers(n_samples: int, population_mean: float, population_std: float) -> float:
    """
    Demonstrate the Law of Large Numbers by computing the sample mean.
    
    Args:
        n_samples: Total number of samples to draw from the distribution
        population_mean: The true mean of the population distribution
        population_std: The true standard deviation of the population distribution
    
    Returns:
        The sample mean
    """
    samples = np.random.normal(population_mean, population_std, n_samples)
    return round(np.mean(samples), 4)
