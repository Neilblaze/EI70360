import math
import numpy as np
from collections import Counter


def compute_tf(doc):
    tf = Counter(doc)
    total_terms = len(doc)
    for term in tf:
        tf[term] /= total_terms
    return tf


def compute_idf(docs):
    n_docs = len(docs)
    idf = {}
    doc_count = Counter(term for doc in docs for term in set(doc))

    for term, count in doc_count.items():
        idf[term] = math.log((n_docs + 1) / (count + 1)) + 1

    return idf


def compute_tf_idf(corpus, query):
    idf = compute_idf(corpus)
    query_term = query[0]
    result = []

    for i, doc in enumerate(corpus):
        tf = compute_tf(doc)
        sub_result = []
        for query_term in query:
            tfidf = tf.get(query_term, 0) * idf.get(query_term, 0)
            sub_result.append(tfidf)

        result.append(sub_result)

    return result


corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "chased", "the", "cat"],
    ["the", "bird", "flew", "over", "the", "mat"],
]
query = ["cat", "mat"]

print(compute_tf_idf(corpus, query))
