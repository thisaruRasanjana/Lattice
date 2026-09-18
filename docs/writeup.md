# Lattice: Deconstructing Parallel Compute

Lattice is a systems engineering project designed to deconstruct the hardware layers beneath modern Large Language Models (LLMs). Instead of relying on high-level frameworks like PyTorch, this project investigates parallel architecture from first principles.

The journey starts with Conway’s Game of Life—a simple cellular automaton—implemented across three paradigms: Sequential CPU, Multi-threaded CPU, and GPU Compute (via Apple's Metal). By profiling these implementations, we uncover the physical bottlenecks of the M2 chip. We then apply those exact lessons to hand-write the foundational kernels of an LLM (Matrix Multiplication and Attention) and ultimately profile a real inference stack to prove our theories.

---

## 1. The Compute Pipeline

### The Evolution of Execution
We built Conway's Game of Life in three phases to understand how hardware scales:
1. **Sequential CPU:** A single core updates a grid of cells. It is smart but isolated.
2. **Multi-threaded CPU:** The grid is partitioned horizontally using `std::thread::scope`. 8 cores operate in parallel without synchronization overhead, providing near-linear scaling up to the core count.
3. **GPU Compute (Metal):** The grid is flattened and passed to the GPU using unified memory (`storageModeShared`). A compute shader maps a single lightweight thread to every individual cell on the grid simultaneously.

### Memory-Bound vs. Compute-Bound
The critical insight of this project emerged in Phase 4 and 5.

**Game of Life is Memory-Bound.**
To update a single cell, the GPU must read 9 neighboring cells from RAM but only perform a few trivial additions. The GPU's Arithmetic Logic Units (ALUs) starve while waiting for data. We observed performance flatlining as the memory traffic hit the M2's unified memory bandwidth ceiling (~100 GB/s).

**Matrix Multiplication is Compute-Bound.**
In Phase 5, we wrote a naive GPU matrix multiplication kernel. Multiplying an $N \times N$ matrix requires $O(N^2)$ memory reads but $O(N^3)$ mathematical operations. Because the ratio of math-to-memory scales exponentially with matrix size, the ALUs finally get saturated. In our benchmarks, the GPU achieved **254 GFLOPS**, completely eclipsing the CPU because it was no longer waiting on RAM.

### The Attention Mechanism
In Phase 6, we implemented the core of a Transformer model: Self-Attention.
We broke it down into three dispatches:
1. `QK^T` (Dot product of Queries and Keys)
2. `Softmax` (A row-wise GPU reduction)
3. `Scores * V` (Weighted sum with Values)

Our naive implementation highlighted the **Quadratic Cost of Attention** ($O(N^2)$). Between steps 1 and 2, the massive `Scores` matrix must be written to global GPU RAM, only to be immediately read back for the Softmax reduction. This forces a compute-bound operation to become fatally memory-bound. This bottleneck is exactly what modern fused architectures (like **FlashAttention**) were invented to solve by keeping the `Scores` matrix entirely within the GPU's ultra-fast SRAM cache.

---

## 2. From Toy Kernels to Real Inference

In Phase 7, we stepped out of Rust and used Apple's `mlx-lm` framework to profile a real local LLM (`Qwen2.5-0.5B-Instruct-4bit` and `1.5B`) on the M2 GPU. The telemetry perfectly mirrored our toy kernels.

### Prefill = Matrix Multiplication (Compute-Bound)
During the "Prefill" phase, the model processes the user's prompt. 
For the 0.5B model, generating the first token for a 50-word prompt took **123.2 ms**. Processing a 150-word prompt (3x larger) took **130.1 ms**. 

Why doesn't tripling the input triple the time? Because the prefill phase bundles all input tokens into a single massive Matrix Multiplication (Phase 5). The GPU thrives on massive workloads to hide memory latency. Tripling the workload simply gave the thousands of idle ALUs something to do.

### Decode = Bandwidth Wall (Memory-Bound)
During "Decode", the model generates one word at a time. 
- 0.5B Model Speed: **~207 tokens/sec**
- 1.5B Model Speed: **~84 tokens/sec**

The 1.5B model generates tokens 2.5x slower. Why? Because decoding a single token requires the GPU to load the *entire* model's parameters from RAM to compute just *one* word (Phase 4). There is almost no math happening relative to the massive amount of data being moved. The ALUs starve, and we crash into the 100 GB/s memory bandwidth wall. More parameters = more data to drag across the bus = slower decode.

### The KV-Cache (Quadratic Attention)
During generation, we observed the 0.5B model's memory footprint grow by ~10MB over 500 tokens. This is the physical manifestation of Phase 6. Every time a new token is generated, its Key and Value vectors are permanently appended to the Attention matrix in RAM (the KV-Cache). As the sequence length $N$ grows, this matrix grows quadratically, threatening to consume all available memory.

---

## 3. Key Design Decisions

- **Flat Buffer Layout:** Decided in Phase 1, storing the 2D grid as a 1D `Vec<u8>` avoided pointer indirection, kept CPU caches hot, and provided a zero-cost abstraction when porting directly to Metal `MTLBuffer`s.
- **`storageModeShared`:** A deliberate, Apple Silicon-specific optimization taking advantage of unified memory to avoid explicit CPU-to-GPU data copies.
- **Staged Attention Kernel:** The Phase 6 attention kernel was intentionally left naive (non-fused) to demonstrate the exact memory traffic jam that FlashAttention was built to eliminate.
- **Leaving Rust for Phase 7:** The goal was to understand a production serving stack, not to prove everything can be done in one language. Using `mlx` allowed us to profile real model architectures built specifically for Apple Silicon.

## 4. Future Explorations

Having established the baseline bottlenecks, future iterations of this project could explore:
- **Tiled Matrix Multiplication:** Using `threadgroup` (shared) memory to cache blocks of matrices, drastically reducing global RAM access.
- **Fused Attention:** Writing a custom FlashAttention-style kernel to compute Softmax on-the-fly without materializing the $O(N^2)$ `Scores` matrix in global memory.
- **Multi-Head Attention:** Extending the toy kernel to support batched, multi-head configurations.
