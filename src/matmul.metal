#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device const float* A  [[buffer(0)]],    // M × K
    device const float* B  [[buffer(1)]],    // K × N
    device float* C        [[buffer(2)]],    // M × N
    constant uint3& dims   [[buffer(3)]],    // (M, K, N)
    uint2 gid              [[thread_position_in_grid]]
) {
    uint col = gid.x;  // which col of C
    uint row = gid.y;  // which row of C
    
    uint M = dims.x;
    uint K = dims.y;
    uint N = dims.z;
    
    // Bounds check
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
