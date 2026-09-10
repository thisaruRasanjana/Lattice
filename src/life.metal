#include <metal_stdlib>
using namespace metal;

kernel void game_of_life(
    device const uint8_t* current [[buffer(0)]],
    device uint8_t* next         [[buffer(1)]],
    constant uint2& grid_size    [[buffer(2)]],
    uint2 gid                    [[thread_position_in_grid]]
) {
    uint width = grid_size.x;
    uint height = grid_size.y;

    // Bounds check — if the GPU launches extra threads beyond the grid, exit early
    if (gid.x >= width || gid.y >= height) return;

    // Count live neighbors (toroidal wrap using modulo)
    uint count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            uint nx = (gid.x + dx + width) % width;
            uint ny = (gid.y + dy + height) % height;
            count += current[ny * width + nx];
        }
    }

    // Apply B3/S23 rules
    uint index = gid.y * width + gid.x;
    uint8_t alive = current[index];
    if (alive == 1) {
        next[index] = (count == 2 || count == 3) ? 1 : 0;
    } else {
        next[index] = (count == 3) ? 1 : 0;
    }
}
