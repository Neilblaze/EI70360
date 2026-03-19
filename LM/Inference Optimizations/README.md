## LLM Inference Optimizations

* **KV cache reuse** -- During autoregressive decoding, attention keys and values from prior tokens are stored so they do not need to be recomputed at each step. This is foundational to all modern transformer inference and turns O(n^2) per-step work into O(n).

* **Paged attention** -- Manages the KV cache in non-contiguous memory pages (similar to OS virtual memory) instead of requiring one large contiguous allocation. Pioneered by vLLM, this eliminates memory fragmentation and enables near-zero waste when serving variable-length sequences.

* **Continuous batching** -- Instead of waiting for an entire batch to finish before starting a new one, requests are inserted and retired from the batch dynamically as they arrive and complete. This keeps the GPU busy and significantly improves throughput compared to static batching.

* **Static batching** -- Groups input sequences of the same (or padded-to-same) length into a single batch so they can be processed together in one forward pass. Simpler than continuous batching but effective when inputs are uniform, maximizing GPU utilization by avoiding warp divergence.

* **Speculative decoding** -- A small, fast "draft" model generates candidate tokens, which are then verified in parallel by the large target model. Accepted tokens skip full-model autoregressive steps, yielding speedups of 2-3x while preserving the exact output distribution of the large model.

* **Early exit / adaptive compute** -- Allows the model to produce an output from an intermediate layer when an internal confidence metric is met, skipping the remaining layers. This reduces per-token latency for "easy" tokens while still using full depth for harder predictions.

* **Token streaming** -- Returns generated tokens to the client one at a time (or in small chunks) as they are produced, rather than waiting for the full response. Dramatically reduces time-to-first-token (TTFT) and perceived latency for end users.

* **Prefix caching** -- Caches and reuses the KV states of shared prompt prefixes (e.g., system prompts) across multiple requests. Avoids redundant prefill computation when many users share the same initial context, common in chat and API serving scenarios.

* **Quantization (INT8/INT4/FP8)** -- Reduces the numerical precision of model weights (and optionally activations) from FP16/BF16 to lower bit-widths. This shrinks model memory footprint, increases throughput by fitting more into cache/registers, and leverages dedicated low-precision tensor cores on modern GPUs.

* **Weight pruning** -- Removes (zeroes out) weights with small magnitudes or low importance scores, producing a sparser model. Reduces both memory footprint and compute, though unstructured sparsity requires hardware or kernel support to realize speed gains.

* **Structured sparsity** -- Enforces sparsity in regular patterns (e.g., 2:4 sparsity where 2 out of every 4 weights are zero) that map directly to hardware acceleration. NVIDIA Ampere+ GPUs have native 2:4 sparse tensor core support, providing up to 2x speedup with minimal accuracy loss.

* **Operator fusion** -- Combines multiple sequential operations (e.g., matmul + bias + activation) into a single GPU kernel launch, reducing kernel launch overhead, intermediate memory reads/writes, and global memory bandwidth consumption.

* **FlashAttention** -- Rewrites the attention computation to operate in tiles that stay in GPU SRAM, avoiding materializing the full N x N attention matrix in HBM. Provides both memory savings (O(N) instead of O(N^2)) and wall-clock speedup by reducing memory bandwidth bottlenecks.

* **Tensor parallelism** -- Splits individual weight matrices across multiple GPUs so that each GPU computes a slice of every layer. Requires all-reduce communication each layer but keeps all GPUs active on every token, making it the standard parallelism strategy for latency-sensitive single-request inference.

* **Pipeline parallelism** -- Assigns different groups of layers to different GPUs, forming a pipeline. While one device processes layer group N for request A, another processes layer group N+1 for request B. Increases throughput for batched inference at the cost of "pipeline bubble" idle time.

* **Sequence parallelism** -- Distributes the sequence (token) dimension across devices for operations like LayerNorm and dropout that are not parallelized by tensor parallelism. Reduces per-device activation memory and complements tensor parallelism in long-context scenarios.

* **Expert parallelism (MoE)** -- In Mixture-of-Experts models, different experts reside on different devices and only the top-k experts (typically 1-2) are activated per token. This keeps total compute per token low while allowing the model to have a much larger total parameter count.

* **Mixed precision inference** -- Runs the model in FP16 or BF16 instead of FP32, halving memory usage and doubling throughput on tensor cores. BF16 is preferred for its wider dynamic range (same exponent bits as FP32), reducing the risk of overflow without loss scaling.

* **Graph compilation (TensorRT, XLA, torch.compile)** -- Traces or captures the model's computation graph ahead of time and applies global optimizations like operator fusion, constant folding, layout transformations, and kernel selection. Produces an optimized executable that runs significantly faster than eager-mode interpretation.

* **Kernel autotuning** -- Benchmarks multiple candidate GPU kernel implementations for each operation (varying tile sizes, block dimensions, etc.) and selects the fastest for the given hardware and input shape. Frameworks like Triton and cuDNN use this to adapt to specific GPU architectures.

* **Memory offloading (CPU/NVMe)** -- Moves inactive model weights, KV cache entries, or optimizer states from GPU HBM to CPU DRAM or NVMe storage. Enables running models larger than GPU memory at the cost of higher latency when data must be fetched back.

* **Activation recomputation trade-offs** -- Instead of storing all intermediate activations (high memory), recomputes them during the backward/forward pass as needed. Common in training, but also used during long-context inference prefill to reduce peak GPU memory.

* **Chunked prefill** -- Splits a long input prompt into smaller chunks and processes them one chunk at a time during the prefill phase. Prevents the prefill of one long prompt from blocking decode steps for other requests, reducing latency spikes in serving.

* **Sliding window attention** -- Restricts each token's attention to only the most recent W tokens rather than the full sequence. Reduces attention compute and KV cache from O(N) to O(W), enabling processing of very long sequences with bounded memory. Used in Mistral and similar models.

* **RoPE scaling / ALiBi** -- Techniques to extend a model's effective context length beyond its training window. RoPE scaling (NTK-aware, YaRN) interpolates rotary position embeddings; ALiBi adds linear bias penalties, enabling extrapolation to longer contexts without retraining.

* **Efficient tokenization pipelines** -- Parallelizes tokenization across CPU threads and/or caches tokenized results for repeated prompts. Prevents the tokenizer from becoming a bottleneck, especially when serving many short requests at high throughput.

* **Request prioritization / scheduling** -- Assigns priorities to incoming requests and schedules them to balance latency targets (e.g., p50, p99) against overall throughput. Ensures latency-sensitive requests (e.g., interactive chat) are served before batch workloads.

* **Warm starts / model preloading** -- Keeps the model weights loaded in GPU memory (and optionally a pre-warmed KV cache or CUDA graph) so that the first request does not incur cold-start latency. Critical for serverless and auto-scaled inference endpoints.

* **LoRA / adapter merging at inference** -- Low-Rank Adaptation (LoRA) adds small trainable matrices to frozen base weights. At inference time, these can be merged into the base weights (zero overhead) or applied on-the-fly, enabling efficient multi-tenant serving with per-user fine-tunes.

* **Distillation for inference models** -- Trains a smaller "student" model to mimic the outputs of a larger "teacher" model. The student runs faster and uses less memory while retaining much of the teacher's quality, making it a direct inference speed optimization.

* **Caching full responses** -- Stores the complete generated response for a given input so that identical future queries can be served instantly from cache without running the model. Effective for FAQ-like workloads or repeated API calls with deterministic sampling.

* **Hardware-aware placement** -- Maps model layers, experts, or KV cache shards to specific GPUs or accelerators based on their compute/memory profiles. Ensures memory-bound layers land on bandwidth-rich devices and compute-bound layers on FLOPS-rich ones.

* **Inference on specialized hardware (TPUs, NPUs)** -- Runs inference on accelerators specifically designed for matrix-heavy workloads. TPUs, Groq LPUs, and Intel Gaudi each offer different throughput-latency characterizations optimized for transformer inference.

* **Asynchronous execution pipelines** -- Overlaps data preprocessing, host-to-device transfers, GPU compute, and device-to-host transfers using separate CUDA streams or equivalent. Hides transfer latency behind compute and keeps the GPU fully utilized.

* **KV cache quantization** -- Compresses the keys and values stored in the attention cache to lower precision (e.g., FP8, INT8, or INT4). This reduces per-token memory footprint and memory bandwidth during attention, allowing larger batch sizes or longer contexts with minimal quality loss.

* **Block-sparse attention** -- Divides the attention matrix into blocks and computes attention only for selected non-zero blocks, skipping the rest. Reduces attention complexity for long sequences when combined with fixed patterns (local + global) or learned sparsity masks.

* **Attention head pruning** -- Permanently removes attention heads that contribute minimally to output quality (measured by importance scores or gradient-based metrics). Reduces both compute and memory proportionally to the fraction of heads removed.

* **Grouped-query attention (GQA)** -- Uses fewer key-value heads than query heads by grouping multiple query heads to share a single KV pair. Introduced in Llama 2 70B, GQA provides a middle ground between full multi-head attention and multi-query attention, cutting KV cache size proportionally.

* **Multi-query attention (MQA)** -- An extreme variant of GQA where all query heads share a single set of keys and values per layer. Dramatically reduces KV cache memory and bandwidth, making it highly effective for long-context, high-batch serving.

* **Weight tying** -- Shares the same weight matrix between the input embedding layer and the output projection (LM head). Reduces parameter count and memory footprint without quality loss, as these two matrices learn similar representations.

* **Vocabulary pruning** -- Restricts the output vocabulary to a relevant subset for domain-specific tasks (e.g., code generation, constrained grammars). Shrinks the softmax computation and output projection, saving compute proportional to the vocabulary reduction.

* **Fused sampling kernels** -- Combines top-k filtering, top-p (nucleus) filtering, temperature scaling, and categorical sampling into a single GPU kernel. Avoids multiple kernel launches and intermediate memory allocations during the sampling phase.

* **Constrained decoding** -- Restricts token generation to follow a predefined grammar, schema, or set of valid tokens at each step (e.g., JSON mode). Reduces the effective search space and can be implemented with logit masking for zero additional compute.

* **Decode-maximal batching** -- Keeps the decode batch as full as possible at every step by immediately inserting new requests as old ones finish. Maximizes GPU utilization during the memory-bandwidth-bound decode phase.

* **Batch-level KV sharing** -- When multiple requests in a batch share the same prefix (e.g., identical system prompt), their KV cache for that prefix is stored once and referenced by all. Saves memory proportional to (num_sharing_requests - 1) times the prefix length.

* **Hierarchical caching** -- Implements a multi-tier cache hierarchy (GPU HBM, CPU DRAM, NVMe SSD) for KV states. Hot entries stay in GPU memory; warm entries spill to CPU; cold entries go to disk, with prefetching to hide retrieval latency.

* **Context compression** -- Compresses or summarizes long conversation histories into a shorter representation before feeding them into the model for the next turn. Reduces prefill compute and KV cache size while retaining essential context.

* **Input packing** -- Packs multiple short sequences into a single long sequence (with attention masking to prevent cross-contamination) to fill the model's context window. Eliminates padding waste and increases GPU utilization when inputs vary in length.

* **Early stopping in decoding** -- Terminates generation before reaching the maximum token limit when an end-of-sequence token is produced or a confidence threshold is met. Saves compute on responses that naturally end early.

* **Speculative sampling trees** -- Extends speculative decoding by having the draft model generate a tree of candidate continuations (multiple branches) rather than a single sequence. The target model verifies the tree in one forward pass, increasing the expected number of accepted tokens.

* **Dual-model cascades** -- Routes queries through a small model first; only queries the small model cannot answer confidently are escalated to a larger, more expensive model. Reduces average inference cost while maintaining quality on hard inputs.

* **Confidence-based routing** -- Uses an uncertainty or entropy estimate from the model (or a lightweight classifier) to decide whether a query needs a large model or can be handled by a smaller/cheaper one. A generalization of cascading applied at the request level.

* **GPU memory defragmentation** -- Periodically compacts or reorganizes GPU memory allocations (especially KV cache blocks) to reclaim fragmented free space. Important in long-running serving systems where allocation/deallocation cycles create unusable memory gaps.

* **Pinned memory usage** -- Allocates host (CPU) memory as page-locked ("pinned"), enabling faster DMA transfers between CPU and GPU. Eliminates an extra copy through a staging buffer, reducing data transfer latency for offloading and prefetching.

* **RDMA / high-speed interconnects** -- Uses Remote Direct Memory Access (via InfiniBand or RoCE) for GPU-to-GPU communication across nodes, bypassing the CPU and OS kernel. Critical for multi-node tensor/pipeline parallel inference where communication latency directly impacts token latency.

* **On-demand expert loading (MoE)** -- In Mixture-of-Experts models, loads expert weights from CPU or disk into GPU memory only when a token is routed to that expert, rather than keeping all experts resident. Enables serving very large MoE models on limited GPU memory.

* **Quantization-aware kernels** -- Custom GPU kernels that operate directly on quantized (INT4/INT8) weights without first dequantizing to FP16. Avoids the memory and compute overhead of explicit dequantization and fully exploits low-precision tensor core throughput.

* **Dequantization fusion** -- Fuses the dequantization step (converting INT4/INT8 weights back to FP16) with the subsequent matrix multiplication into a single kernel. Eliminates an extra memory read/write pass and reduces kernel launch overhead.

* **Operator reordering** -- Rearranges the execution order of independent operations to improve memory access locality and reduce peak memory usage. For example, reordering attention and FFN sub-operations can reduce the number of costly global memory round-trips.

* **NUMA-aware placement** -- On multi-socket CPU systems, pins data and computation to the same NUMA (Non-Uniform Memory Access) node to avoid expensive cross-socket memory accesses. Relevant when CPU is used for part of inference (e.g., tokenization, embedding lookup, or hybrid execution).

* **Async tokenization + inference** -- Overlaps tokenization of the next request with GPU inference on the current request, so the CPU tokenizer and GPU model run concurrently. Prevents tokenization from becoming a serial bottleneck in high-throughput pipelines.

* **Prefill-decode disaggregation** -- Assigns prefill (prompt processing) and decode (token generation) to separate hardware pools. Prefill is compute-bound and benefits from high-FLOPS devices; decode is memory-bandwidth-bound and benefits from high-bandwidth setups. Splitwise and DistServe implement this pattern.

* **Model surgery (layer dropping)** -- Removes the last N transformer layers from a trained model and uses the remaining layers for inference. Often the final layers contribute diminishing returns, so dropping them trades a small quality loss for proportional latency and memory savings.

* **Low-rank KV cache approximation** -- Approximates the stored KV cache entries using low-rank matrix factorization, compressing the cache without quantization. Reduces memory proportionally to the chosen rank while maintaining attention quality.

* **Hybrid CPU-GPU execution** -- Offloads lightweight operations (embedding lookup, sampling, post-processing) to the CPU while the GPU handles the heavy matmuls and attention. Frees GPU cycles for compute-critical work and can improve overall throughput.

* **Cold-start mitigation (snapshotting)** -- Saves a fully initialized model state (weights loaded, CUDA contexts ready, KV cache pre-allocated) as a snapshot that can be restored in milliseconds. Eliminates multi-second cold-start delays in serverless and auto-scaled deployments.

* **Adaptive precision scaling** -- Assigns different numerical precisions to different layers or operations based on their sensitivity. Insensitive layers run in INT4/INT8 for speed; sensitive layers keep FP16/BF16 for accuracy, giving a better speed-quality tradeoff than uniform quantization.

* **Dynamic layer skipping** -- Skips entire transformer layers for tokens deemed "easy" by a lightweight gating mechanism or confidence check. Similar to early exit but applied selectively per-layer rather than halting at a fixed point.

* **Mixture-of-depths** -- Dynamically varies how many transformer layers each token passes through, using a learned routing mechanism. Easy tokens traverse fewer layers; hard tokens use the full depth, reducing average per-token compute.

* **KV cache sharding across GPUs** -- Distributes KV cache entries across multiple GPUs to avoid a single device bottleneck. Each GPU holds a portion of the sequence's KV cache, and attention is computed in a distributed fashion (as in ring attention).

* **KV cache preallocation** -- Reserves a fixed block of GPU memory for KV cache at model load time, avoiding repeated dynamic allocations during serving. Prevents runtime memory fragmentation and allocation overhead, especially under high request concurrency.

* **Chunked decoding** -- Generates tokens in small blocks (e.g., 4-8 tokens) rather than strictly one at a time, enabling better GPU utilization by increasing the arithmetic intensity of each decode step. Complements speculative decoding and parallel decoding techniques.

* **Parallel decoding heads** -- Attaches multiple output heads to the model to predict several future tokens simultaneously in a single forward pass. Used by Medusa and similar methods; accepted tokens advance the sequence while rejected ones are discarded.

* **Lookahead decoding** -- Generates multiple future token candidates in parallel using Jacobi-style fixed-point iteration, without a separate draft model. Tokens that converge to stable values are accepted, reducing the number of serial autoregressive steps.

* **Draft model distillation** -- Trains the draft model in speculative decoding specifically to match the target model's token distribution, maximizing acceptance rate. A better-aligned draft means fewer rejected tokens and higher end-to-end speedup.

* **Semantic caching** -- Uses embedding similarity (rather than exact string matching) to identify cache-worthy queries, returning cached responses for semantically equivalent inputs. More robust than exact caching but requires a similarity threshold to avoid stale hits.

* **Stateful sessions** -- Persists the KV cache and conversation context server-side between turns of a multi-turn conversation. Eliminates re-processing the entire conversation history on each turn, significantly reducing prefill cost in chat applications.

* **Incremental prompt updates** -- When a user appends to an existing prompt, only the new tokens are processed through prefill, reusing the cached KV states for the existing prefix. Avoids redundant computation proportional to the unchanged prefix length.

* **Multi-resolution attention** -- Applies fine-grained attention to recent/nearby tokens and coarser attention to distant tokens. Reduces the cost of long-range attention while preserving local precision. Variants include dilated attention and hierarchical attention patterns.

* **Adaptive attention span** -- Allows each attention head or layer to learn its own effective context window length. Heads that only need local context automatically use a short span (less compute); heads that need global context use the full sequence.

* **Recurrent memory tokens** -- Compresses past context into a fixed number of special "memory" tokens that are prepended to the current input. The model attends to these compressed tokens instead of the full history, bounding context processing cost.

* **Throughput-oriented decoding modes** -- Configures the serving system to prioritize aggregate tokens-per-second over individual request latency. Involves larger batches, longer batching windows, and potentially reordering requests to maximize GPU utilization.

* **Latency-oriented decoding modes** -- Configures the system to minimize time-to-first-token and per-token latency for individual requests, even at the cost of lower overall throughput. Uses smaller batches, immediate scheduling, and prioritized compute allocation.

* **Tail-latency mitigation** -- Specifically targets the slowest requests (p95/p99) through techniques like request hedging, preemption of long-running requests, or reserving capacity for stragglers. Ensures consistent user experience across all requests.

* **Backpressure control** -- Prevents the inference server from accepting more requests than it can handle, applying throttling or queuing when load exceeds capacity. Avoids cascading latency degradation and out-of-memory crashes under traffic spikes.

* **Admission control** -- Rejects or enqueues incoming requests when the system is at capacity, returning appropriate error codes (e.g., 429) rather than degrading performance for all in-flight requests. A critical component of production serving systems.

* **Autoscaling policies** -- Automatically adjusts the number of serving replicas based on metrics like queue depth, GPU utilization, or request latency. Scales out during traffic spikes and scales in during lulls to balance cost and performance.

* **Model variant routing** -- Selects which model (e.g., 7B vs 70B) to use for a given query based on estimated difficulty, required quality, or cost constraints. Reduces average cost by routing easy queries to cheaper models.

* **Quantized KV attention kernels** -- Specialized GPU kernels that perform the attention computation directly on quantized (FP8/INT8) key-value tensors. Avoids dequantizing KV cache entries before attention and reduces memory bandwidth consumption.

* **Fused rotary embedding kernels** -- Integrates the computation of rotary position embeddings (RoPE) into the attention kernel itself, rather than applying it as a separate operation. Eliminates an extra memory read/write pass over the Q and K tensors.

* **Custom CUDA kernels** -- Hand-written GPU kernels optimized for specific operations in the inference pipeline (e.g., attention, sampling, or quantized matmul). Outperform auto-generated or generic kernels by exploiting hardware-specific features.

* **Compiler-level graph rewrites** -- Optimization passes applied by compilers (TensorRT, XLA, Inductor) that automatically detect and eliminate redundant operations, fold constants, and simplify the computation graph before execution.

* **Activation sparsification** -- Zeroes out activations below a threshold during inference, reducing the number of non-zero values that subsequent layers must process. Effective when combined with sparse kernels that can skip zero-valued computations.

* **Dynamic neuron gating** -- Uses a lightweight gating network to selectively activate only a subset of neurons (or channels) in each feed-forward layer per token. Reduces compute proportionally to the fraction of deactivated neurons.

* **Non-autoregressive decoding (NAR)** -- Generates all output tokens in parallel in one or a few forward passes, rather than one token at a time. Trades some output quality for dramatic speedups, useful for tasks like translation where quality loss is acceptable.

* **Mask-predict decoding** -- An iterative NAR approach that generates all tokens at once, then masks low-confidence tokens and re-predicts them over several rounds. Converges to high-quality output in far fewer iterations than fully autoregressive decoding.

* **KV reuse across beams** -- In beam search, beams sharing the same prefix also share the corresponding KV cache entries rather than duplicating them. Reduces KV cache memory by up to (beam_width - 1)x for shared prefixes.

* **On-the-fly model compression** -- Compresses model weights at load time (e.g., quantizing from FP16 to INT4) rather than pre-converting the checkpoint. Allows a single stored checkpoint to be served at different precision levels without maintaining multiple copies.

* **Prefill-decode disaggregation at scale (Splitwise)** -- Dedicates separate GPU clusters to the prefill and decode phases of inference, connected by a high-speed network for KV cache transfer. Each cluster is hardware-optimized for its phase (compute-heavy vs bandwidth-heavy).

* **State space models (Mamba/SSM) inference** -- Runs inference with O(N) complexity using a fixed recurrent state instead of an attention mechanism. No KV cache is needed; each new token updates a fixed-size state, making very long sequences feasible with constant memory.

* **Linear attention / retention networks (RetNet, RWKV)** -- Reformulates attention as a linear recurrence, achieving O(1) per-token cost during autoregressive generation. Trades some expressiveness for dramatically cheaper long-sequence inference.

* **Hybrid attention-SSM architectures (Jamba, Zamba)** -- Interleaves standard transformer attention layers with SSM (Mamba) layers in a single model. Attention layers handle tasks requiring precise token-to-token relationships; SSM layers handle long-range context cheaply.

* **KV cache disaggregation (remote KV stores)** -- Offloads KV cache to a dedicated distributed memory pool (e.g., Mooncake, MemServe) separate from the compute GPUs. Decouples memory capacity from compute, enabling larger effective caches.

* **Attention sinks** -- Retains a small set of initial "sink" tokens (typically the first few tokens) permanently in the KV cache during streaming inference. Stabilizes attention distributions in models that allocate disproportionate attention to the first tokens, as shown in StreamingLLM.

* **Prompt caching (API-level)** -- Exposes prefix caching as a first-class feature in an inference API, allowing users to mark reusable prompt sections. The server maintains cached KV states across requests, reducing both latency and per-token billing for shared prefixes.

* **Medusa decoding** -- Attaches multiple lightweight prediction heads to the base model, each predicting a different future token position in parallel. A tree-based verification step accepts consistent multi-token predictions in a single forward pass.

* **EAGLE / EAGLE-2 decoding** -- Creates draft predictions at the feature (hidden state) level rather than the token level, using a small autoregressive head on top of the target model's features. Achieves higher acceptance rates than token-level speculative decoding.

* **Jacobi decoding** -- Treats autoregressive generation as a fixed-point iteration problem, initializing all output positions and iteratively refining them in parallel until convergence. Reduces the number of serial steps needed, especially for predictable sequences.

* **Self-speculative decoding** -- Uses the same model's early layers (via early exit) as the draft model, eliminating the need for a separate smaller model. Reduces serving complexity and memory overhead while still providing speculative speedups.

* **GPTQ / AWQ / SmoothQuant** -- Widely deployed post-training quantization algorithms. GPTQ uses approximate second-order information for weight-only quantization; AWQ preserves salient weights; SmoothQuant migrates quantization difficulty from activations to weights.

* **1-bit / binary quantization (BitNet)** -- Quantizes model weights to 1.58 bits (ternary: -1, 0, +1), replacing floating-point matmuls with additions and subtractions. Dramatically reduces memory and enables extremely fast inference on hardware supporting integer/popcount operations.

* **Cross-layer attention (CLA)** -- Shares KV cache entries across multiple transformer layers (e.g., layers 1 and 2 share the same KV) to reduce total cache memory. Requires architectural support during training but provides significant memory savings at inference.

* **Sarathi-Serve / chunked-prefill scheduling** -- Interleaves chunks of prefill computation for new requests with ongoing decode steps for existing requests. Prevents a single long-prompt prefill from stalling all decode-phase requests, improving tail latency.

* **Multi-LoRA serving (S-LoRA, Punica)** -- Serves many different LoRA adapters concurrently from a single base model by batching the LoRA computations using custom kernels (e.g., SGMV in Punica). Avoids duplicating the full base model for each adapter.

* **Ring attention** -- Distributes long-sequence attention across devices arranged in a ring, where each device computes attention for a chunk of the sequence and passes KV blocks to the next device. Enables context lengths beyond 1M tokens.

* **Sparse attention patterns (Longformer, BigBird)** -- Uses a combination of local sliding window attention and sparse global attention tokens to reduce attention complexity from O(N^2) to O(N). Enables efficient inference on very long documents.

* **Cross-attention caching (encoder-decoder)** -- In encoder-decoder models (T5, Whisper), the encoder output is computed once and its cross-attention KV states are cached and reused across all decoder steps. Eliminates redundant encoder computation during generation.

* **Sliding window + global token hybrid (Mistral-style)** -- Combines local sliding window attention (each token attends to the nearest W tokens) with a small number of global attention tokens that attend to the full sequence. Balances efficiency with the ability to propagate long-range information.

* **Cascade / waterfall inference (FrugalGPT)** -- Routes each query through progressively larger models, stopping as soon as a model produces a sufficiently confident answer. Reduces average cost by handling easy queries with small models.

* **FlashAttention-3 / FlashDecoding++** -- Updated variants of FlashAttention optimized for newer GPU architectures (H100/Hopper). Uses asynchronous warp specialization, overlapped softmax with GEMM, and FP8 support for further speedups over FlashAttention-2.

* **NVLink / NVSwitch topology-aware placement** -- Arranges tensor parallel shards and KV cache partitions according to the physical GPU interconnect topology. Places communicating GPUs on high-bandwidth NVLink connections and avoids slow PCIe hops.

* **FP8 end-to-end pipeline (H100 native)** -- Runs the entire inference pipeline (including matmuls, attention, and normalization) in FP8 using Hopper tensor cores natively. Distinct from post-hoc quantization because the model is trained or calibrated for FP8 from the start.

* **Chain-of-thought length control** -- Limits or compresses the intermediate reasoning tokens in chain-of-thought models (o1, R1-style) to reduce total token budget without eliminating reasoning capability. Directly reduces inference cost for reasoning-heavy models.

* **Process reward model (PRM)-guided search** -- Uses a process reward model to score intermediate reasoning steps during tree search at inference time. Prunes low-quality branches early, reducing the total number of forward passes needed to find a good solution.

* **Monte Carlo Tree Search (MCTS) for inference** -- Applies MCTS during generation for tasks like math reasoning or code, exploring multiple solution paths and using rollout evaluations to guide search. A test-time compute scaling strategy that trades latency for quality.

* **Best-of-N sampling with reward model scoring** -- Generates N independent completions, scores each with a reward model, and returns the highest-scoring one. A simple but effective test-time compute strategy where quality scales with N at linear cost.

* **Visual token compression (LLaVA-style)** -- Reduces the number of image patch tokens before feeding them into the language model backbone in vision-language models. Techniques include token merging, pooling, or learned downsampling, directly reducing prefill compute.

* **Speculative decoding for multimodal models** -- Adapts the speculative decoding paradigm to vision-language or audio-language pipelines, using a smaller multimodal draft model or a text-only draft after the vision encoder. Reduces autoregressive decode latency for multimodal generation.

* **Tool call batching** -- Issues multiple tool or function calls in a single round-trip when the model generates several independent tool invocations. Reduces total wall-clock time compared to sequential execution of each tool call.

* **Parallel tool execution** -- Executes independent tool calls concurrently when there are no data dependencies between them. Combined with tool call batching, this minimizes the latency overhead of multi-tool agentic workflows.

* **Depth-first vs breadth-first beam search** -- Depth-first beam search fully expands one beam before moving to the next, reducing peak memory by not holding all partial beams simultaneously. Breadth-first expands all beams level by level, offering better throughput but higher memory.