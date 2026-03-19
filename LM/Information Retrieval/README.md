# Information Retrieval Models & Methods

## 1. Set-Theoretic / Early Retrieval Models

* **Boolean Retrieval Model** — The earliest and most straightforward retrieval model, where documents are retrieved if they satisfy a Boolean expression over index terms using AND, OR, and NOT operators. It provides no ranking; results are either matched or not, which limits its effectiveness for large-scale retrieval since users must formulate precise queries.

* **Extended Boolean Model (p-norm model)** — Introduced by Salton, Fox, and Wu (1983), this model generalizes the strict Boolean model by assigning partial match scores using the p-norm distance formula. By tuning the parameter p, the model interpolates between a pure Boolean match (p -> infinity) and a vector-space-like soft match (p = 1), enabling ranked output from Boolean-style queries.

* **Fuzzy Retrieval Model** — Applies fuzzy set theory (Zadeh, 1965) to information retrieval, where term membership in a document is represented as a degree in [0, 1] rather than a binary value. This allows the model to handle partial or imprecise matches, capturing the inherent vagueness and uncertainty in natural language queries and document representations.

* **Proximity / Positional Retrieval** — Retrieves and ranks documents based on the distance between query terms within the document text, favoring documents where the terms appear close together. Positional indexes store term positions, enabling operators such as "within k words" or ordered/unordered proximity constraints, which significantly improve precision over bag-of-words models.

* **Phrase Retrieval** — Requires that query terms appear as an exact contiguous phrase in the document, rather than as independent terms scattered throughout the text. Phrase indexes or positional indexes with bigram/trigram indexing are used to support this, and phrase matching is a common default in web search engines when users enclose queries in quotation marks.

* **Wildcard / Truncation Retrieval** — Supports queries containing wildcard characters (e.g., `*`, `?`) that match partial terms, enabling morphological or exploratory search without requiring the user to know exact terms. Implemented using data structures such as permuterm indexes, k-gram indexes, or tries, it is especially useful for handling spelling variations, morphological variants, and prefix/suffix-based searches.

* **Zone / Field-Based Retrieval** — Retrieves and scores documents by matching queries against specific document zones or fields (e.g., title, abstract, body, author), with each zone potentially receiving a different weight. This is critical in structured or semi-structured document collections, where relevance signals from the title field, for instance, are typically far stronger than those from the body.

---

## 2. Vector Space & Algebraic Models

* **Vector Space Model (Salton, 1975)** — One of the foundational models in IR, developed by Gerard Salton and colleagues for the SMART system. Documents and queries are represented as high-dimensional term vectors, and relevance is measured by the cosine of the angle between them. Its simplicity and effectiveness established it as the dominant retrieval framework for decades, though it assumes term independence.

* **TF-IDF Weighting** — A term-weighting scheme that combines term frequency (TF), measuring how often a term appears in a document, with inverse document frequency (IDF), measuring how rare the term is across the corpus. This product gives high weight to terms that are frequent in a specific document but uncommon overall, effectively balancing local importance with global discriminative power. It remains the most widely used unsupervised term-weighting scheme.

* **SMART Weighting Scheme** — A family of TF-IDF variants developed for the SMART Information Retrieval System at Cornell, parameterized by a three-character notation (e.g., `lnc.ltc`) specifying the TF component, IDF component, and normalization for both document and query vectors. This framework systematized the comparison of different weighting strategies and heavily influenced the design of modern retrieval systems.

* **Latent Semantic Indexing (LSI / LSA)** — Uses truncated Singular Value Decomposition (SVD) on the term-document matrix to discover latent semantic structure, mapping documents and queries into a lower-dimensional space where synonymy and polysemy effects are partially resolved. Introduced by Deerwester et al. (1990), LSI was one of the earliest methods to move beyond exact lexical matching, though its computational cost and difficulty in handling new documents limited large-scale adoption.

* **Generalized Vector Space Model** — Proposed by Wong, Ziarko, and Wong (1985), this model relaxes the orthogonality (independence) assumption between index terms in the standard VSM. By introducing correlations between term vectors, it captures term co-occurrence relationships, allowing documents sharing semantically related but lexically different terms to be recognized as similar.

* **Topic-Based Vector Space Model (TVSM)** — Represents documents through distributions over discovered topics rather than raw term vectors, typically using topic models such as Latent Dirichlet Allocation (LDA) or Probabilistic LSA (pLSA). This reduces dimensionality and captures thematic content, enabling retrieval based on topical similarity rather than surface-level term overlap.

* **Random Indexing** — A computationally efficient alternative to SVD-based dimensionality reduction, where terms are assigned sparse random vectors and document representations are accumulated by summing the random vectors of their constituent terms. Introduced by Kanerva et al. (2000), it approximates the results of LSI at a fraction of the computational cost, making it suitable for large-scale and incremental settings.

* **Non-Negative Matrix Factorization (NMF)** — Decomposes the term-document matrix into two non-negative factor matrices, yielding an additive parts-based representation of documents. Unlike SVD, the non-negativity constraint produces interpretable components that correspond to coherent topics or themes, making NMF useful for both retrieval and topic discovery in document collections.

---

## 3. Probabilistic Retrieval Models

* **Binary Independence Model (BIM)** — One of the earliest probabilistic retrieval models, proposed by Robertson and Sparck Jones (1976), which estimates the probability of relevance based on the presence or absence of terms, assuming that terms are independently distributed in relevant and non-relevant documents. Despite its strong independence assumption, BIM provided the theoretical foundation for the Probability Ranking Principle and later models like BM25.

* **Okapi BM25** — Developed at City University London by Robertson, Walker, and colleagues in the early 1990s as part of the Okapi system, BM25 is a probabilistic ranking function that incorporates term frequency saturation (preventing over-counting of repeated terms) and document length normalization. It consistently outperforms simpler TF-IDF schemes and remains one of the most effective and widely deployed unsupervised retrieval functions in both research benchmarks and production search engines.

* **BM25+ / BM25L** — Refinements of BM25 that address its known bias against long documents and its problematic behavior when term frequency is zero. BM25+ (Lv and Zhai, 2011) adds a small positive constant to the TF component to ensure that matching documents always score higher than non-matching ones, while BM25L uses a different TF normalization formula to achieve a similar correction for document length bias.

* **BM25F** — An extension of BM25 designed for structured documents with multiple fields (e.g., title, body, anchor text), where term frequencies from different fields are combined with field-specific weights and saturation parameters before computing the final score. This avoids the suboptimal practice of scoring each field independently and summing, and it is used extensively in web search engines.

* **Two-Poisson Model** — Models term frequency distributions in a collection using a mixture of two Poisson distributions: one for "elite" terms (content-bearing terms that appear frequently in relevant documents) and one for non-elite terms. Proposed by Harter (1975) and further developed by Robertson and Walker, this model provides a principled basis for TF saturation effects seen in BM25.

* **Language Modeling for IR (LMIR)** — Introduced by Ponte and Croft (1998), this framework estimates a unigram language model for each document and ranks documents by the likelihood that the query was generated by the document's model. Smoothing the document model (e.g., using Jelinek-Mercer, Dirichlet, or Absolute Discount smoothing) is essential to handle unseen terms and acts as an implicit form of IDF weighting.

* **Query Likelihood Model** — A specific instantiation of the language modeling approach, where documents are ranked by P(Q|D), the probability of generating the query Q from the document language model D. Its elegance lies in avoiding the need to explicitly model relevance; instead, the generative probability serves as a proxy for topical relevance, with smoothing methods playing a critical role in retrieval effectiveness.

* **KL-Divergence Retrieval Model** — Rather than computing the likelihood of the query given a document model, this approach (Lafferty and Zhai, 2001) ranks documents by the Kullback-Leibler divergence between the query language model and the document language model. This formulation naturally supports relevance feedback and query expansion by modifying the query model, and provides a symmetric information-theoretic framework for retrieval.

* **Divergence From Randomness (DFR)** — A family of probabilistic weighting models proposed by Amati and Van Rijsbergen (2002), based on the idea that informative terms deviate significantly from a random distribution. DFR decomposes the term weight into an "informativeness" component (measuring divergence from a basic randomness model such as Poisson or geometric) and a "gain" component (measuring risk of accepting a term based on its elite status), offering a principled and modular framework for term weighting.

* **Inference Network Model** — Proposed by Turtle and Croft (1991), this model represents the retrieval process as a Bayesian inference network with document and query nodes connected through concept representation nodes. Belief propagation through the network computes a belief score for each document, providing a flexible framework that can incorporate multiple evidence sources, including term occurrences, phrases, and passage-level information.

* **Belief Network Retrieval Model** — A broader class of probabilistic graphical models applied to IR, where relevance is inferred through evidential reasoning over a network of random variables representing documents, queries, and concepts. It generalizes the inference network model and supports complex dependency structures, though computational tractability requires approximations for large-scale retrieval settings.

* **Dirichlet Smoothing** — A widely used smoothing method for language models in IR, where the document model is smoothed with the collection model using a Dirichlet prior parameterized by a single hyperparameter mu. Proposed by Zhai and Lafferty (2001), Dirichlet smoothing is length-adaptive: shorter documents receive more smoothing (relying more on collection statistics), while longer documents rely more on their own term frequencies. It is often the default smoothing method in language model-based retrieval systems.

---

## 4. Relevance Feedback & Query Reformulation

* **Rocchio Algorithm** — The classical relevance feedback algorithm for the Vector Space Model, proposed by Rocchio (1971), which adjusts the query vector by moving it toward the centroid of known relevant documents and away from the centroid of known non-relevant documents. Controlled by alpha, beta, and gamma parameters, it remains the foundational method for explicit relevance feedback and is simple to implement in practice.

* **Pseudo Relevance Feedback (PRF)** — Also known as blind relevance feedback, this technique assumes that the top-k documents from an initial retrieval are relevant and uses them as implicit feedback to expand or reweight the query. PRF is widely used because it requires no user interaction, and it consistently improves recall at the cost of occasionally introducing drift when the initial results are poor.

* **Blind Feedback** — A term often used interchangeably with Pseudo Relevance Feedback, referring to automatic query expansion using terms extracted from the top-ranked documents of an initial retrieval, without any explicit user judgment. While effective on average across queries, blind feedback can degrade performance for queries where the initial retrieval results are off-topic, a problem known as query drift.

* **Local Context Analysis (LCA)** — A query expansion technique proposed by Xu and Croft (1996) that identifies expansion terms by analyzing term co-occurrence patterns within the top-retrieved documents. Unlike global methods, LCA adapts expansion terms to the specific query context, combining the advantages of local feedback with the robustness of corpus-level co-occurrence statistics.

* **Global Analysis Expansion** — Uses corpus-wide statistics, such as term co-occurrence matrices, thesauri, or distributional similarity measures, to identify expansion terms independently of any retrieval results. Methods include using mutual information, chi-squared statistics, or pre-built semantic resources like WordNet, providing consistent expansion at the cost of not being tailored to the query context.

* **Query Reformulation** — A broad category of techniques that iteratively modify or restructure the query to improve retrieval effectiveness, encompassing both automatic methods (term reweighting, expansion, reduction) and interactive approaches (suggesting alternative queries to users). Modern search engines perform extensive query reformulation including spelling correction, synonym expansion, and query segmentation before executing the retrieval.

* **Relevance Models (RM1 / RM3)** — Probabilistic query expansion models introduced by Lavrenko and Croft (2001), where RM1 estimates a relevance language model from pseudo-relevant documents by computing a weighted combination of document language models. RM3 extends RM1 by interpolating the estimated relevance model with the original query model, providing a controlled balance between expansion and the original query that consistently yields state-of-the-art unsupervised retrieval performance.

* **Interactive Relevance Feedback** — A retrieval paradigm where the user explicitly marks retrieved documents as relevant or non-relevant, and the system uses these judgments to iteratively refine the query. Unlike pseudo relevance feedback, interactive feedback relies on genuine human assessment, producing more accurate query modifications but requiring user effort, which limits its applicability in many real-world settings.

---

## 5. Graph-Based Retrieval

* **PageRank** — Developed by Brin and Page (1998) for the Google search engine, PageRank computes a global importance score for each web page based on the link structure of the web, modeling a random surfer who follows links and occasionally jumps to a random page. The stationary distribution of this random walk gives each page a score reflecting the quantity and quality of incoming links, and it remains a foundational concept in web search ranking.

* **HITS (Hyperlink-Induced Topic Search)** — Proposed by Kleinberg (1999), HITS is a query-dependent link analysis algorithm that computes two mutually reinforcing scores for each page: a hub score (how well it points to authoritative pages) and an authority score (how well it is pointed to by good hubs). Unlike PageRank, HITS is computed at query time over a subgraph of relevant pages, making it more topically focused but also more susceptible to manipulation.

* **SALSA (Stochastic Approach for Link-Structure Analysis)** — A link analysis algorithm proposed by Lempel and Moran (2001) that combines aspects of both PageRank and HITS by performing random walks on a bipartite graph of hubs and authorities. SALSA is less susceptible to the topic drift problem of HITS and more robust against link spam, while still computing distinct hub and authority scores.

* **Topic-Sensitive PageRank** — Proposed by Haveliwala (2002), this extension of PageRank computes multiple PageRank vectors, each biased toward a different topic category, rather than a single global rank. At query time, the appropriate topic-specific PageRank vector is selected based on the query's topic, providing a form of personalized or context-dependent ranking that improves over generic PageRank.

* **Random Walk Retrieval Models** — A general class of retrieval models based on random walk processes over document graphs, where edges may represent hyperlinks, citation relationships, or content similarity. These models propagate relevance information through the graph structure, supporting semi-supervised retrieval scenarios and enabling the incorporation of relational evidence beyond traditional content-based features.

* **Knowledge Graph-Based Retrieval** — Leverages structured knowledge graphs (e.g., Wikidata, Freebase, or domain-specific ontologies) to enhance retrieval by incorporating entity relationships, type hierarchies, and semantic constraints. Entity linking maps query and document mentions to knowledge graph entities, enabling semantic matching and entity-centric retrieval that goes beyond lexical surface forms.

---

## 6. Learning-to-Rank Methods

* **Pointwise Learning-to-Rank** — Treats ranking as a regression or classification problem, where a model predicts the relevance score or class of each document independently, without considering other documents in the result list. Methods include linear regression, logistic regression, and ordinal regression applied to feature vectors of query-document pairs; however, pointwise approaches do not directly optimize ranking metrics like NDCG or MAP.

* **Pairwise Learning-to-Rank (RankNet)** — Formulates ranking as a binary classification problem over pairs of documents, where the model learns to predict which document in a pair is more relevant. RankNet, introduced by Burges et al. (2005) at Microsoft Research, uses a neural network with a cross-entropy loss defined over pairwise preferences, and it was one of the first successful applications of neural networks to learning-to-rank in web search.

* **Listwise Learning-to-Rank (LambdaRank / LambdaMART)** — Directly optimizes ranking quality metrics (such as NDCG or ERR) over entire ranked lists, rather than individual documents or pairs. LambdaRank defines implicit gradients (lambdas) proportional to the change in NDCG from swapping two documents, while LambdaMART combines these lambda gradients with gradient boosted regression trees, achieving state-of-the-art effectiveness in many benchmark evaluations and commercial search engines.

* **RankSVM** — A pairwise learning-to-rank method proposed by Joachims (2002) that applies Support Vector Machines to learn a ranking function by maximizing the margin between correctly ordered document pairs. It formulates ranking as a large-margin classification problem on preference pairs derived from relevance judgments, and its convex optimization guarantees and strong performance on smaller feature sets made it influential in the early LTR literature.

* **Gradient Boosted Ranking Trees (GBRT)** — Ensemble methods based on gradient boosted decision trees (e.g., XGBoost, LightGBM) applied to ranking tasks, which iteratively build regression trees to minimize a ranking-oriented loss function. GBRT-based approaches, particularly LambdaMART, have dominated learning-to-rank benchmarks and are the backbone of ranking in most modern commercial search engines due to their ability to capture complex feature interactions.

* **ListNet** — A listwise learning-to-rank method proposed by Cao et al. (2007) that defines a probability distribution over permutations of documents using top-one probabilities (derived from the Plackett-Luce model) and minimizes the cross-entropy between the ground-truth permutation distribution and the predicted one. ListNet was among the first approaches to formalize listwise ranking loss and directly optimize over the entire ranked list structure.

* **AdaRank** — A listwise learning-to-rank algorithm proposed by Xu and Li (2007) that directly optimizes performance measures such as MAP or NDCG using a boosting framework. At each round, AdaRank selects a weak ranker and updates query weights to focus on queries that are currently poorly ranked, making it an intuitive and effective method for directly targeting ranking metric improvement.

* **Coordinate Ascent** — A linear learning-to-rank method proposed by Metzler and Croft (2007) that optimizes a ranking metric (e.g., MAP) by iteratively tuning one feature weight at a time while holding all others fixed. Despite its simplicity, coordinate ascent is competitive with more complex methods, easy to implement, and interpretable, making it widely used as a baseline and in production systems via the RankLib library.

---

## 7. Neural Retrieval (Dense & Sparse)

* **Dense Passage Retrieval (DPR)** — Introduced by Karpukhin et al. (2020) at Facebook AI, DPR uses a dual-encoder architecture with two independent BERT encoders (one for queries, one for passages) to produce dense vector representations, where inner-product similarity replaces lexical matching. Trained on question-answer pairs with hard negative mining, DPR demonstrated that dense retrieval can outperform BM25 for open-domain question answering when sufficient training data is available.

* **Bi-Encoder Retrieval** — A general class of dense retrieval models where separate encoder networks independently map queries and documents into a shared embedding space, allowing document embeddings to be precomputed and retrieved efficiently via Approximate Nearest Neighbor (ANN) search at query time. Bi-encoders are the primary architecture for scalable dense retrieval, though they sacrifice some accuracy compared to cross-encoders because the query and document do not interact during encoding.

* **Cross-Encoder Reranking** — Jointly encodes the query and document as a single concatenated input through a transformer model (e.g., BERT), enabling deep token-level interaction between query and document terms for precise relevance scoring. While cross-encoders achieve superior accuracy over bi-encoders, their computational cost (quadratic in sequence length) makes them impractical for first-stage retrieval over millions of documents, so they are typically used as rerankers over a smaller candidate set.

* **Late Interaction Models (ColBERT)** — Introduced by Khattab and Zaharia (2020), ColBERT computes contextualized token-level embeddings independently for queries and documents using BERT, then estimates relevance via a "MaxSim" operation that finds, for each query token, the maximum cosine similarity with any document token. This late-interaction design preserves much of the effectiveness of cross-encoders while enabling precomputation of document token embeddings and efficient retrieval through ANN indexes.

* **Sparse Neural Retrieval (SPLADE)** — A transformer-based sparse retrieval model that produces high-dimensional sparse representations over the vocabulary, using the MLM (masked language model) head of a pretrained transformer to predict term importance weights, including expansion terms not present in the original text. SPLADE (Formal et al., 2021) combines neural relevance learning with the efficiency of inverted index infrastructure, achieving competitive or superior performance to dense retrieval on many benchmarks.

* **DeepCT** — A neural term reweighting model proposed by Dai and Callan (2019) that uses BERT to predict context-aware term importance scores for each term in a passage, replacing static TF values in the inverted index. By producing more accurate term weights that reflect contextual meaning, DeepCT improves the effectiveness of traditional inverted index retrieval without requiring any change to the search infrastructure.

* **uniCOIL** — A single-vector contextualized exact-match retrieval model that assigns a scalar weight to each token in a document or query using a BERT-based architecture, producing a sparse representation where only tokens appearing in the text receive non-zero weights. Proposed by Lin and Ma (2021), uniCOIL bridges dense and sparse retrieval by providing BERT-quality term weighting within the standard inverted index framework.

* **ANCE (Approximate Nearest Neighbor Negative Contrastive Estimation)** — A dense retrieval training strategy proposed by Xiong et al. (2020) that uses an asynchronously updated ANN index to select hard negative documents during training, rather than relying on random or in-batch negatives. This produces substantially better dense representations by exposing the model to the most confusing negatives, addressing a key limitation of earlier dense retrieval training methods.

* **doc2query / docTTTTTquery** — A document expansion technique where a sequence-to-sequence neural model (e.g., T5) is trained to generate synthetic queries that a document might answer, and these generated queries are appended to the document before indexing. This bridges the vocabulary mismatch between queries and documents at the index level, improving recall for traditional sparse retrieval systems like BM25 without modifying the retrieval algorithm itself.

* **Sentence-BERT (SBERT)** — Introduced by Reimers and Gurevych (2019), SBERT fine-tunes a siamese BERT network using a contrastive or triplet loss to produce semantically meaningful fixed-size sentence embeddings suitable for efficient similarity search. SBERT embeddings are widely used as a general-purpose retrieval backbone for semantic search, clustering, and paraphrase detection tasks.

---

## 8. Hybrid Retrieval Approaches

* **Dense + Sparse Hybrid Retrieval** — Combines dense vector-based retrieval (capturing semantic similarity) with sparse lexical retrieval (capturing exact term matches) to exploit the complementary strengths of both paradigms. In practice, a BM25 or SPLADE retrieval provides high-recall lexical coverage, while a dense bi-encoder captures semantic relationships, and their results are merged to produce a final ranking that is typically more robust than either method alone.

* **Reciprocal Rank Fusion (RRF)** — A simple yet highly effective unsupervised rank combination method proposed by Cormack, Clarke, and Butt (2009) that fuses ranked lists from multiple retrieval systems by assigning each document a score based on the reciprocal of its rank in each list (1 / (k + rank)). RRF requires no training or parameter tuning (except for the constant k, typically set to 60), and consistently performs well across diverse fusion scenarios.

* **Weighted Score Fusion** — Combines retrieval scores from multiple systems or features using a linear combination with learned or tuned weights, producing a unified score for each document. This approach requires score normalization (e.g., min-max or z-score normalization) across systems to ensure comparability, and the weights can be optimized using held-out relevance judgments to maximize a target ranking metric.

* **Cascade Retrieval Pipelines** — A multi-stage retrieval architecture where a fast, high-recall first-stage retriever (e.g., BM25 or a bi-encoder) produces a broad candidate set, which is then progressively refined by increasingly expensive but more accurate rerankers (e.g., cross-encoders). This telescoping design balances computational cost with retrieval effectiveness, and modern search systems typically employ two to four stages.

* **Approximate Nearest Neighbor Retrieval (ANN)** — Algorithms and data structures (e.g., HNSW, IVF, product quantization, ScaNN) that enable sub-linear-time similarity search in high-dimensional embedding spaces, trading a small amount of recall for orders-of-magnitude speed improvements over exact search. ANN is the critical infrastructure that makes dense retrieval practical at scale, and libraries such as FAISS, ScaNN, and Annoy are widely used in production systems.

* **Cross-Encoder Distillation into Bi-Encoders** — A training strategy where a high-accuracy cross-encoder teacher model generates soft relevance labels (scores) for query-document pairs, which are then used to train a more efficient bi-encoder student model through knowledge distillation. This technique, used in systems like TAS-B and ColBERTv2, allows bi-encoders to approach cross-encoder accuracy while retaining their scalability advantage for first-stage retrieval.

---

## 9. Diversity & Result Selection Methods

* **Maximal Marginal Relevance (MMR)** — Introduced by Carbonell and Goldstein (1998), MMR is a greedy re-ranking algorithm that iteratively selects documents by balancing relevance to the query with novelty relative to already-selected documents, controlled by a lambda parameter. It is the most widely used diversification technique in practice and has been adopted well beyond IR, including in RAG pipelines and summarization systems, to reduce redundancy in output.

* **xQuAD (Explicit Query Aspect Diversification)** — A probabilistic diversification framework proposed by Santos, Macdonald, and Ounis (2010) that explicitly models query subtopics (aspects) and selects documents to maximize the coverage of these aspects while minimizing redundancy. xQuAD estimates the probability that each document satisfies each subtopic and uses this to re-rank the result list, making it particularly effective when query ambiguity or multi-faceted information needs are present.

* **IA-Select (Intent-Aware Select)** — A diversification algorithm introduced by Agrawal et al. (2009) that models different possible user intents behind an ambiguous query and selects documents to maximize the probability that at least one relevant document for each intent appears in the top results. IA-Select uses a probabilistic framework where candidate documents are evaluated based on how well they cover underrepresented intents in the current selection.

* **Subtopic Retrieval Models** — A class of retrieval and evaluation approaches designed to ensure that search results cover the multiple subtopics or facets of a broad or ambiguous query, rather than focusing on a single dominant interpretation. Evaluation metrics like alpha-NDCG and ERR-IA are specifically designed to measure subtopic coverage, and these models are central to the TREC Web Track diversity task.

* **PM-2 (Proportionality Model 2)** — A proportional diversification algorithm proposed by Dang and Croft (2012) that selects documents to ensure the representation of different query aspects in the result list is proportional to their estimated importance. Inspired by proportional representation in political science, PM-2 avoids the winner-take-all behavior of greedy methods, producing more balanced coverage across subtopics.

---

## 10. Multimodal & Specialized Retrieval

* **Cross-Modal Retrieval** — Retrieves items across different modalities (e.g., text-to-image, image-to-text, text-to-video), typically by learning a shared embedding space where representations from different modalities can be directly compared. Models like CLIP (Radford et al., 2021) and ALIGN have made cross-modal retrieval practical at scale by training on massive paired datasets, enabling applications such as visual search and video retrieval.

* **Content-Based Image Retrieval (CBIR)** — Retrieves images based on visual features extracted from the image content itself (color histograms, texture descriptors, deep CNN features) rather than relying on text metadata or tags. Modern CBIR systems use deep learning features from pretrained convolutional or vision transformer networks, achieving retrieval accuracy that far surpasses traditional hand-crafted feature approaches.

* **Entity Retrieval** — Focuses on retrieving structured entities (e.g., people, organizations, products) from knowledge bases or entity-enriched document collections in response to entity-bearing queries. Entity retrieval is fundamental to search engine knowledge panels, question answering over knowledge bases, and recommendation systems, and it often involves entity linking, type filtering, and entity-oriented language models.

* **Passage Retrieval** — Retrieval at the paragraph, sentence, or span level rather than the full-document level, enabling more precise answers to specific information needs, especially in question answering and conversational search. Passage retrieval requires document segmentation strategies and passage-level indexing, and it has become the dominant retrieval granularity in open-domain question answering systems.

* **Conversational Retrieval** — Retrieval conditioned on the context of an ongoing multi-turn dialogue, where the current query must be interpreted in light of the previous conversation history. Techniques include query rewriting (resolving coreferences and ellipsis from dialogue context), conversational query understanding, and contextual re-ranking, as studied in the TREC CAsT (Conversational Assistance) track.

* **Structured Document Retrieval (XML / JSON Retrieval)** — Retrieves not entire documents but specific structural elements (e.g., XML elements, sections, or subsections) from semi-structured document collections, as studied in the INEX evaluation campaigns. This requires specialized indexing and scoring that respects document structure, and the challenge lies in selecting the most appropriate granularity of response: not too broad and not too specific.

---

## 11. Modern LLM-Era Retrieval Techniques

* **Retrieval-Augmented Generation (RAG)** — A paradigm introduced by Lewis et al. (2020) that combines a retrieval step with a generative language model: the retriever fetches relevant documents from a knowledge corpus, and the generator conditions on both the query and the retrieved documents to produce the final output. RAG addresses the knowledge cutoff and hallucination limitations of standalone LLMs by grounding generation in retrieved evidence, and it has become the dominant architecture for knowledge-intensive NLP tasks.

* **Iterative Retrieval** — A multi-step retrieval strategy where the system performs multiple rounds of retrieval, using the output of previous retrieval and reasoning steps to formulate new queries, enabling complex multi-hop information gathering. Architectures like IRCoT (Trivedi et al., 2023) interleave chain-of-thought reasoning with retrieval, and systems like ITER-RETGEN iterate between retrieval and generation to progressively refine both the retrieved context and the generated answer.

* **Self-Query Retrieval** — A technique where an LLM analyzes the user's natural language query and converts it into structured retrieval filters (metadata filters, date ranges, category constraints) combined with a semantic search component. This enables retrieval over structured and semi-structured knowledge bases without requiring users to know the underlying schema, and it is implemented in frameworks like LangChain for RAG applications.

* **Multi-Vector Retrieval** — Uses multiple embedding vectors per document to capture different aspects, passages, or semantic facets, rather than compressing the entire document into a single vector. ColBERT's per-token embedding approach is the most prominent example, but multi-vector methods also include segment-level embeddings and hierarchical representations that preserve fine-grained information lost in single-vector compression.

* **Chunked Retrieval** — Splits documents into smaller semantic units (chunks) of fixed or variable length before indexing, so that retrieval operates at the chunk level rather than over entire documents. Chunking strategy (size, overlap, boundary detection) significantly impacts retrieval quality: overly large chunks dilute relevance signals while overly small chunks lose context. Sophisticated approaches use sentence boundaries, paragraph breaks, or recursive splitting with overlap.

* **Query Decomposition Retrieval** — Breaks a complex, multi-faceted query into multiple simpler sub-queries, retrieves documents for each sub-query independently, and then aggregates or synthesizes the results. Approaches range from LLM-driven decomposition (e.g., using chain-of-thought prompting to generate sub-questions) to formal query analysis methods, and this technique is particularly important for handling compositional and multi-hop questions in RAG systems.

* **Hypothetical Document Embeddings (HyDE)** — Proposed by Gao et al. (2022), HyDE uses an LLM to generate a hypothetical document that would answer the query, then encodes this hypothetical document using a dense encoder and uses the resulting embedding for retrieval. This bridges the query-document asymmetry problem by searching with a document-like representation, and it has been shown to improve zero-shot dense retrieval without any relevance-labeled training data.

* **Generative Retrieval (DSI / GENRE)** — A paradigm where a single sequence-to-sequence model directly maps queries to document identifiers (docids), bypassing the traditional index-retrieve pipeline entirely. The Differentiable Search Index (DSI) by Tay et al. (2022) memorizes document contents and identifier mappings in model parameters, while GENRE (De Cao et al., 2021) generates entity names autoregressively using constrained beam search. This emerging paradigm challenges the separation between indexing and retrieval but currently faces scalability limitations.

* **Instruction-Tuned Retrieval Embeddings (E5 / GTE / BGE)** — A family of modern embedding models (e.g., E5-Mistral, GTE, BGE) that are fine-tuned with natural language task instructions prepended to queries, enabling a single model to produce embeddings optimized for different retrieval tasks (e.g., "Retrieve passages that answer this question" vs. "Find similar documents"). These models, trained on large-scale curated datasets with contrastive learning, currently achieve state-of-the-art results on retrieval benchmarks such as MTEB.