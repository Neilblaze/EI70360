import math
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1
    """
    if not candidate:
        return 0.0

    candidate_len = len(candidate)

    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - candidate_len), x))

    brevity_penalty = 1.0
    if candidate_len < closest_ref_len:
        brevity_penalty = math.exp(1 - closest_ref_len / candidate_len)

    precisions = []

    for n in range(1, max_n + 1):
        cand_ngrams = Counter()
        for i in range(len(candidate) - n + 1):
            cand_ngrams[tuple(candidate[i:i+n])] += 1

        if not cand_ngrams:
            return 0.0

        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter()
            for i in range(len(ref) - n + 1):
                ref_ngrams[tuple(ref[i:i+n])] += 1

            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

        clipped_count = 0
        for ngram, cand_count in cand_ngrams.items():
            ref_count = max_ref_counts.get(ngram, 0)
            clipped_count += min(cand_count, ref_count)

        precision = clipped_count / sum(cand_ngrams.values())
        precisions.append(precision)

    if any(p == 0 for p in precisions):
        return 0.0

    log_sum = sum(math.log(p) for p in precisions)
    geometric_mean = math.exp(log_sum / max_n)
    return brevity_penalty * geometric_mean
