# Information Retrieval Models & Methods

## 1. Set-Theoretic / Early Retrieval Models

* **Boolean Retrieval Model** — Exact matching using AND, OR, NOT operators.
* **Extended Boolean Model (p-norm model)** — Soft Boolean ranking using weighted term contributions.
* **Fuzzy Retrieval Model** — Uses fuzzy set theory to handle uncertainty in term membership.
* **Proximity / Positional Retrieval** — Retrieves documents based on term distance constraints.
* **Phrase Retrieval** — Requires exact phrase matches instead of independent terms.
* **Wildcard / Truncation Retrieval** — Supports partial word matching using patterns.

---

## 2. Vector Space & Algebraic Models

* **Vector Space Model (Salton 1975)** — Represents documents and queries as vectors for cosine similarity.
* **TF–IDF weighting** — Term weighting based on frequency and corpus rarity.
* **SMART weighting scheme** — Family of TF-IDF variants used in the SMART IR system.
* **Latent Semantic Indexing (LSI/LSA)** — Uses SVD to capture latent term-document relationships.
* **Generalized Vector Space Model** — Removes independence assumption between index terms.
* **Topic-based Vector Space Model (TVSM)** — Represents documents via topic distributions.
* **Random Indexing** — Dimensionality reduction alternative to SVD for semantic space.

---

## 3. Probabilistic Retrieval Models

* **Binary Independence Model (BIM)** — Probability of relevance based on term occurrence independence.
* **Okapi BM25** — State-of-the-art probabilistic ranking function using TF saturation and length normalization.
* **BM25+ / BM25L** — Improvements to BM25 for long-document bias correction.
* **BM25F** — Extension of BM25 for multi-field documents.
* **Two-Poisson Model** — Models term frequency distribution with two Poisson distributions.
* **Language Modeling for IR (LMIR)** — Treats documents as language models generating queries.
* **Query Likelihood Model** — Computes probability of query given document model.
* **KL-Divergence Retrieval Model** — Uses divergence between query and document language models.
* **Divergence From Randomness (DFR)** — Ranks by how much term occurrence deviates from randomness.
* **Inference Network Model** — Bayesian network representation of the retrieval process.
* **Belief Network Retrieval Model** — Probabilistic network capturing evidential reasoning.

---

## 4. Relevance Feedback & Query Reformulation

* **Rocchio Algorithm** — Adjusts query vectors using relevant and non-relevant document centroids.
* **Pseudo Relevance Feedback (PRF)** — Uses top retrieved documents as assumed relevant feedback.
* **Blind Feedback** — Query expansion without explicit user feedback.
* **Local Context Analysis (LCA)** — Expands queries using term co-occurrence in top documents.
* **Global Analysis Expansion** — Uses corpus-wide statistics for expansion.
* **Query Reformulation** — Iteratively modifies queries based on retrieval results.
* **Relevance Models (RM1/RM3)** — Probabilistic query expansion using language models.

---

## 5. Graph-Based Retrieval

* **PageRank** — Ranks pages based on link structure importance.
* **HITS (Hyperlink-Induced Topic Search)** — Computes hub and authority scores.
* **SALSA** — Random-walk-based link analysis algorithm.
* **Topic-Sensitive PageRank** — Personalized ranking using topic distributions.
* **Random Walk Retrieval Models** — Graph propagation for ranking documents.

---

## 6. Learning-to-Rank Methods

* **Pointwise Learning-to-Rank** — Predicts relevance score independently for each document.
* **Pairwise Learning-to-Rank (RankNet)** — Learns preference between document pairs.
* **Listwise Learning-to-Rank (LambdaRank / LambdaMART)** — Optimizes ranking metrics directly.
* **RankSVM** — Pairwise ranking using support vector machines.
* **Gradient Boosted Ranking Trees (GBRT)** — Tree-based ranking models used in search engines.

---

## 7. Neural Retrieval (Dense & Sparse)

* **Dense Passage Retrieval (DPR)** — Dual-encoder dense embedding retrieval for QA.
* **Bi-Encoder Retrieval** — Independent encoders for query and documents enabling ANN search.
* **Cross-Encoder Reranking** — Jointly encodes query-document pairs for precise relevance scoring.
* **Late Interaction Models (ColBERT)** — Token-level similarity aggregation for efficient ranking.
* **Sparse Neural Retrieval (SPLADE)** — Transformer-based sparse lexical expansion.
* **DeepCT** — Neural term reweighting for inverted-index retrieval.
* **uniCOIL** — Contextualized token weighting for sparse retrieval.

---

## 8. Hybrid Retrieval Approaches

* **Dense + Sparse Hybrid Retrieval** — Combines vector search with lexical retrieval.
* **Reciprocal Rank Fusion (RRF)** — Combines rankings from multiple retrieval systems.
* **Weighted Score Fusion** — Linear combination of retrieval scores.
* **Cascade Retrieval Pipelines** — Multi-stage retrieval (BM25 → neural reranker).
* **Approximate Nearest Neighbor Retrieval (ANN)** — Efficient search in embedding space.

---

## 9. Diversity & Result Selection Methods

* **Maximal Marginal Relevance (MMR)** — Balances relevance with diversity.
* **xQuAD (Explicit Query Aspect Diversification)** — Diversifies results based on query subtopics.
* **IA-Select** — Diversification based on intent-aware ranking.
* **Subtopic Retrieval Models** — Ensure coverage across multiple query aspects.

---

## 10. Multimodal & Specialized Retrieval

* **Cross-Modal Retrieval** — Retrieves across modalities (e.g., text→image).
* **Content-Based Image Retrieval (CBIR)** — Uses visual features instead of text metadata.
* **Entity Retrieval** — Retrieves entities instead of documents.
* **Passage Retrieval** — Retrieval at paragraph or span level.
* **Conversational Retrieval** — Retrieval conditioned on dialogue context.

---

## 11. Modern LLM-Era Retrieval Techniques

* **Retrieval-Augmented Generation (RAG)** — Combines retrieval with generative models.
* **Iterative Retrieval** — Multi-hop retrieval based on previous reasoning steps.
* **Self-Query Retrieval** — LLM converts queries into structured retrieval filters.
* **Multi-Vector Retrieval** — Uses multiple embeddings per document (e.g., ColBERT indexing).
* **Chunked Retrieval** — Splits documents into smaller semantic units for retrieval.
* **Query Decomposition Retrieval** — Breaks complex queries into multiple sub-queries.
