#pragma once
#include <torch/torch.h>

namespace sv {
    
struct MiniCam {
    torch::Tensor c2w;     // (4x4)
    torch::Tensor w2c;     // (4x4)
    int           width  = 0;
    int           height = 0;
    float         fx     = 0.f;
    float         fy     = 0.f;
    float         cx     = 0.f;
    float         cy     = 0.f;
    float         near   = 0.02f;
    std::string   cam_mode = "persp";
    int           frame_id = -1;

    static MiniCam fromIntrinsics(float fx, float fy, float cx, float cy,
                                int width, int height, int frame_id = -1)
    {
        MiniCam cam;
        cam.fx = fx;
        cam.fy = fy;
        cam.cx = cx;
        cam.cy = cy;
        cam.width = width;
        cam.height = height;
        cam.frame_id = frame_id;
        return cam;
    }
};
}
