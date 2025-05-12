#pragma once
#include <torch/torch.h>

namespace sv   // ← everything in sv::
{
struct MiniCam
{
    torch::Tensor c2w;     // (4x4)
    torch::Tensor w2c;     // NEW: add this too
    int           width  = 0;
    int           height = 0;
    float         fovx   = 0.f;
    float         fovy   = 0.f;
    float         near   = 0.02f;
    // Optional: add cx, cy, mode for clarity
    float         cx     = 0.f;
    float         cy     = 0.f;
    std::string   cam_mode = "persp";
    int frame_id;  // <-- Add this!
};
    
} // namespace sv
