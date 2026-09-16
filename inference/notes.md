# Phase 7: Local LLM Profiling on Apple Silicon

We successfully ran Apple's `mlx-lm` framework to profile two local models on the M2 GPU: **Qwen2.5-0.5B** and **Qwen2.5-1.5B**.

These are the results of how the toy kernels we wrote in Phases 5 and 6 scale up to a real serving stack.

## Telemetry

### Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (~310MB RAM)

**1. Time To First Token (Prefill Phase)**
*   Short (10 words): `TTFT = 179.0 ms | Peak Mem = 0.31 GB`
*   Medium (~50 words): `TTFT = 123.2 ms | Peak Mem = 0.32 GB`
*   Long (~150 words): `TTFT = 130.1 ms | Peak Mem = 0.35 GB`

**2. Decode Speed & KV-Cache Growth**
*   Gen 10 tokens: `Speed =  60.5 tok/s | Peak Mem = 0.311 GB`
*   Gen 50 tokens: `Speed = 148.0 tok/s | Peak Mem = 0.311 GB`
*   Gen 100 tokens: `Speed = 176.9 tok/s | Peak Mem = 0.311 GB`
*   Gen 250 tokens: `Speed = 199.2 tok/s | Peak Mem = 0.311 GB`
*   Gen 500 tokens: `Speed = 206.6 tok/s | Peak Mem = 0.311 GB`

### Model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~900MB RAM)

**1. Time To First Token (Prefill Phase)**
*   Short (10 words): `TTFT = 349.2 ms | Peak Mem = 0.90 GB`
*   Medium (~50 words): `TTFT = 182.2 ms | Peak Mem = 0.93 GB`
*   Long (~150 words): `TTFT = 280.3 ms | Peak Mem = 0.96 GB`

**2. Decode Speed & KV-Cache Growth**
*   Gen 10 tokens: `Speed =  35.3 tok/s | Peak Mem = 0.898 GB`
*   Gen 50 tokens: `Speed =  67.8 tok/s | Peak Mem = 0.897 GB`
*   Gen 100 tokens: `Speed =  75.7 tok/s | Peak Mem = 0.897 GB`
*   Gen 250 tokens: `Speed =  81.8 tok/s | Peak Mem = 0.897 GB`
*   Gen 500 tokens: `Speed =  84.2 tok/s | Peak Mem = 0.897 GB`

---

## Analysis: Connecting Theory to Reality

### 1. Prefill is Phase 5 (Compute-Bound Matrix Multiplication)
Notice how for the 0.5B model, generating the first token for a 50-word prompt took `123.2 ms`, and doing it for a 150-word prompt (3x more input) took `130.1 ms` — barely any difference.

Why doesn't tripling the input triple the time? Because of what we learned in **Phase 5**. The prefill phase bundles all the input tokens into a single massive Matrix Multiplication. As we saw when we hit 254 GFLOPS in Phase 5, the GPU thrives on massive matrices because it hides memory latency. The GPU has so many idle workers that tripling the workload didn't actually slow it down; it just gave the idle workers something to do.

Also notice the Short prompt's TTFT (`179 ms`) is *higher* than the Medium prompt (`123 ms`). This is because the very first `generate()` call includes MLX's one-time kernel compilation overhead. Once the Metal shaders are compiled and cached, subsequent calls are faster regardless of prompt length.

### 2. Decode is Phase 4 (Memory-Bound Bandwidth)
Once the first token is generated, the model switches to "Decode" mode, generating one token at a time.
For the 0.5B model, it peaks at `~207 tokens/sec`.
For the 1.5B model, it peaks at `~84 tokens/sec`.

The 1.5B model is 3x larger but generates tokens 2.5x slower. Why? Because of what we learned in **Phase 4** (Game of Life). Decoding a single token requires the GPU to load the *entire* model's parameters from RAM to compute just *one* word. There is almost no math happening relative to the sheer amount of data being moved. The ALUs starve, and you hit the 100 GB/s memory bandwidth wall of the M2 chip. More parameters = more data to drag across the bus = slower decode.

Also notice the "warmup" effect: Gen 10 tokens shows `60.5 tok/s` for the 0.5B model, but by Gen 500 it reaches `206.6 tok/s`. The first few tokens pay the overhead of Metal pipeline setup and KV-cache allocation. Once the pipeline is warm, the decode phase reaches a steady state.

### 3. KV-Cache Growth is Phase 6 (The Quadratic Attention Cost)
Look closely at the peak memory for the 0.5B model during generation:
- Gen 10 tokens: `0.311 GB`
- Gen 500 tokens: `0.311 GB`

At this small scale, the KV-cache growth is too small to observe in the coarse `get_peak_memory()` measurement. Each token adds two vectors of size `d_model` (896 floats for Qwen2.5-0.5B) per attention layer per head. At 4-bit quantization, 500 tokens adds roughly `500 × 24 layers × 896 × 2 × 0.5 bytes ≈ 10 MB`. This is consistent with the numbers — the memory barely budges.

However, this is exactly what we built in **Phase 6**. Every time a new token is generated, its Key and Value vectors are permanently appended to memory (the KV-Cache). Because attention has a quadratic $O(N^2)$ cost, if we kept generating up to 128,000 tokens, the KV-Cache would consume gigabytes of RAM, completely filling up the M2's unified memory.
