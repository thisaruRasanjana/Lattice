#include <metal_stdlib>
using namespace metal;

// 1. Q @ K^T / sqrt(d_k)
kernel void qkt_scaled(
    device const float* Q  [[buffer(0)]],    // seq_len × d_k
    device const float* K  [[buffer(1)]],    // seq_len × d_k
    device float* Scores   [[buffer(2)]],    // seq_len × seq_len
    constant uint2& dims   [[buffer(3)]],    // (seq_len, d_k)
    constant float& scale  [[buffer(4)]],    // Scale factor
    uint2 gid              [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    
    uint seq_len = dims.x;
    uint d_k = dims.y;
    
    if (row >= seq_len || col >= seq_len) return;
    
    float sum = 0.0;
    for (uint i = 0; i < d_k; i++) {
        sum += Q[row * d_k + i] * K[col * d_k + i];
    }
    Scores[row * seq_len + col] = sum * scale;
}

// 2. Row-wise Softmax
kernel void softmax_row(
    device float* matrix   [[buffer(0)]],    // seq_len × seq_len
    constant uint& seq_len [[buffer(1)]],
    uint id                [[thread_position_in_grid]]
) {
    uint row = id;
    
    if (row >= seq_len) return;
    
    uint row_offset = row * seq_len;
    
    // Find max
    float max_val = -INFINITY;
    for (uint i = 0; i < seq_len; i++) {
        max_val = max(max_val, matrix[row_offset + i]);
    }
    
    // Exp and sum
    float sum_exp = 0.0;
    for (uint i = 0; i < seq_len; i++) {
        float e = exp(matrix[row_offset + i] - max_val);
        matrix[row_offset + i] = e;
        sum_exp += e;
    }
    
    // Normalize
    for (uint i = 0; i < seq_len; i++) {
        matrix[row_offset + i] /= sum_exp;
    }
}

// 3. Scores @ V
kernel void matmul_scores_v(
    device const float* Scores [[buffer(0)]], // seq_len × seq_len
    device const float* V      [[buffer(1)]], // seq_len × d_v
    device float* Output       [[buffer(2)]], // seq_len × d_v
    constant uint2& dims       [[buffer(3)]], // (seq_len, d_v)
    uint2 gid                  [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    
    uint seq_len = dims.x;
    uint d_v = dims.y;
    
    if (row >= seq_len || col >= d_v) return;
    
    float sum = 0.0;
    for (uint i = 0; i < seq_len; i++) {
        sum += Scores[row * seq_len + i] * V[i * d_v + col];
    }
    Output[row * d_v + col] = sum;
}
