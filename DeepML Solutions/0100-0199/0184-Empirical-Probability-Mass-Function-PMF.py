def empirical_pmf(samples):
    if not samples:
        return []

    freq = {}
    n = len(samples)
    for sample in samples:
        if sample not in freq:
            freq[sample] = 1 / n
        else:
            freq[sample] += 1 / n

    result = [(k, v) for k, v in freq.items()]
    result = sorted(result, key=lambda x: x[0])
    return result
