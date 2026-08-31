# Lattice

A multi-phase systems engineering project bridging the gap between distributed backend architecture and hardware-level ML infrastructure. 

**Lattice** starts as a raw CPU implementation of Conway's Game of Life in Rust and builds up to GPU-accelerated matrix operations and local LLM inference profiling. The goal is to understand how parallel architectures actually behave at the hardware level—managing memory layouts, cache lines, and compute grids—rather than just calling high-level framework APIs.

## Architecture & Memory Layout (Phase 1)
To optimize for L1/L2 cache locality, Lattice does not use standard 2D arrays (pointers to pointers). 
* **Flat Memory:** The 2D grid is flattened into a continuous 1D `Vec<u8>`. `(x, y)` coordinates are mathematically mapped to the 1D index via `y * width + x`.
* **Double Buffering:** To prevent mid-generation state contamination, the grid allocates two identical buffers (`current` and `next`). The simulation reads from `current`, writes to `next`, and swaps pointers in `O(1)` time via `std::mem::swap`.
* **Boundary Resolution:** Supports both Toroidal (wrap-around via Euclidean modulo arithmetic) and Fixed (hard-wall bounds checking) edge cases.

## The Roadmap

- [x] **Phase 1: Baseline Sequential CPU (Rust)** - Flat memory mapping, double buffering, and correct state updates.
- [ ] **Phase 2: OS-Level CPU Concurrency** - Multithreading via `std::thread`, explicit synchronization (mutex/atomics), and spatial partitioning.
- [ ] **Phase 3: GPU Acceleration (Metal)** - Rewriting the per-cell update as a Metal compute shader (MSL) dispatched from Rust.
- [ ] **Phase 4: Bottleneck Profiling** - Comparing throughput across versions, analyzing memory access patterns, warp occupancy, and latency.
- [ ] **Phase 5: Matrix Multiplication Kernel** - Hand-writing a GPU matmul kernel to bridge grid concurrency with ML primitives.
- [ ] **Phase 6: Toy Attention Kernel** - Implementing $QK^T$, softmax, and weighted sum operations on the GPU.
- [ ] **Phase 7: Local LLM Profiling** - Using Apple's MLX / llama.cpp to profile a real model on M2 (batching impacts, memory bandwidth, KV-cache dynamics).

## Getting Started

Make sure you have the Rust toolchain installed.

Run the test suite:
```bash
cargo test
