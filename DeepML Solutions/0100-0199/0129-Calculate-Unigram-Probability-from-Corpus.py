def unigram_probability(corpus: str, word: str) -> float:
    count = 0
    corpus = corpus.split()
    for i in range(len(corpus)):
        if corpus[i] == word:
            count += 1

    return round(count / len(corpus), 4)
