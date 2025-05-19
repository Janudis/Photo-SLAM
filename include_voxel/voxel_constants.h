#pragma once
namespace sv {
    constexpr int   MAX_OCT_LEVEL      = 7;      // == svraster_cuda.meta.MAX_NUM_LEVELS‑1
    constexpr float MIN_VOX_SIZE       = 0.01f;  // metres – stop subdividing/prune
    constexpr int MAX_VOXEL_LEVEL = 8;  // Same as svraster_cuda.meta.MAX_NUM_LEVELS
}
