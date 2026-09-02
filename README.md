# Lattice

A multi-phase systems engineering project that implements Conway's Game of Life three ways — sequential CPU, multi-threaded CPU, and GPU via Metal — then uses the same GPU infrastructure to hand-write matrix multiplication and attention kernels, ultimately applying that understanding to profiling local LLM inference on Apple Silicon.

The goal is to understand how parallel architectures behave at the hardware level: cache lines, memory layouts, and compute grids, rather than relying on high-level framework abstractions.

**Target:** M2 MacBook Air — Rust for all CPU phases, MSL (Metal Shading Language) for GPU compute.

---

## Roadmap

- [x] **Phase 1 — Sequential CPU:** Flat memory layout, double buffering, and B3/S23 state updates with baseline throughput & bandwidth profiling.
- [x] **Phase 2 — Multi-threaded CPU:** Row-band partitioning via `std::thread::scope`, synchronization-free disjoint writes, multi-core scaling analysis.
- [ ] **Phase 3 — GPU (Metal):** Per-cell update as a Metal compute shader dispatched from Rust. Unified memory via `storageModeShared` with isolated dispatch vs kernel timing.
- [ ] **Phase 4 — Bottleneck Profiling & Analysis:** Comprehensive cross-architecture comparison across grid sweeps, hardware bandwidth saturation, and warp occupancy.
- [ ] **Phase 5 — Matmul Kernel:** Naive GPU matrix multiplication. Each thread computes one output element.
- [ ] **Phase 6 — Toy Attention Kernel:** `QK^T`, row-wise softmax, and weighted sum as three separate dispatches.
- [ ] **Phase 7 — Local LLM Profiling:** MLX / llama.cpp on M2. Time-to-first-token, tokens/sec, KV-cache memory growth.
- [ ] **Phase 8 — Writeup:** Portfolio narrative connecting phase 4 throughput results to phase 7 inference profiling.

---

## Architecture & Performance Model

The grid is stored as a flat `Vec<u8>` of length `width × height`, indexed as `y * width + x`. This avoids the pointer indirection of a nested `Vec<Vec<_>>`, keeps CPU access cache-local, and is directly compatible with an `MTLBuffer` for the GPU phases — no restructuring required when porting to Metal.

Double buffering is used throughout: generation N is read from one buffer, generation N+1 is written to a second, and the two are swapped in O(1) via `std::mem::swap`. This ensures all cells update simultaneously and makes Phase 2 (disjoint thread writes) and Phase 3 (GPU cannot read and write the same buffer in one dispatch) both correct with no special-casing.

### Memory Bandwidth Estimation
Every phase measures throughput (cells/second) and reasons about memory traffic. In Game of Life, updating a cell reads 9 cells (itself + 8 neighbors) and writes 1 cell result (~10 bytes/cell). By calculating `(10 bytes × total_cells) / elapsed_time`, we compute an upper-bound memory bandwidth estimate and compare it against the M2's theoretical peak memory bandwidth (~100 GB/s) to evaluate whether a kernel is compute-bound or memory-bandwidth-bound.

---

## Usage

```bash
# Sequential CPU benchmark (default: 512x512, 100 generations)
cargo run --release -- seq

# Sequential CPU benchmark with custom size and generations
cargo run --release -- seq 1024 500

# Multi-threaded CPU benchmark sweep (tests 1, 2, 4, 8, 10, 16 threads)
cargo run --release -- par sweep

# Multi-threaded CPU benchmark with N threads (default: 4)
cargo run --release -- par 8

# Live random-seed visualization
cargo run -- visual

# Interactive draw mode — paint cells, Space to simulate, R to reset
cargo run -- draw

# Run tests
cargo test
```

---

## Toolchain

- Rust (stable, 2021 edition)
- [`minifb`](https://github.com/emoon/rust_minifb) — raw framebuffer window for visualization
- [`rand`](https://github.com/rust-random/rand) — grid seeding
- [`metal`](https://github.com/gfx-rs/metal-rs) — Rust bindings to Apple's Metal API *(Phase 3+)*
- [`criterion`](https://github.com/bheisler/criterion.rs) — benchmarking harness *(Phase 4)*
