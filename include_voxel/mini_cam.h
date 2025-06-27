#pragma once
#include <torch/torch.h>
#include "include_voxel/voxel_camera.h"

namespace sv {

struct MiniCam {
    torch::Tensor c2w;          // (4×4, float32, CPU)
    torch::Tensor w2c;          // (4×4)
    int   width   = 0;
    int   height  = 0;
    float fx      = 0.f;
    float fy      = 0.f;
    float cx      = 0.f;
    float cy      = 0.f;
    float tanfovx = 1.f;        // = 0.5 * w / fx
    float tanfovy = 1.f;        // = 0.5 * h / fy
    float near    = 0.02f;
    std::string cam_mode = "persp";
    int   frame_id = -1;

    inline static MiniCam fromIntrinsics(float fx, float fy, float cx, float cy,
                                     int w, int h, int frame_id = -1)
    {
        MiniCam m;
        m.width = w;  m.height = h;
        m.fx = fx;    m.fy = fy;
        m.cx = cx;    m.cy = cy;
        m.tanfovx = (fx > 1e-6f) ? 0.5f * w / fx : 1.f;
        m.tanfovy = (fy > 1e-6f) ? 0.5f * h / fy : 1.f;
        // m.tanfovx = std::tan(fx * 0.5f);
        // m.tanfovy = std::tan(fy * 0.5f);

        m.frame_id = frame_id;
        return m;
    }
};

/* -------- helper: build MiniCam from a C++ Camera + pose ---------------- */
inline MiniCam fromCamera(const Camera& cam,
                          const torch::Tensor& c2w,
                          int frame_id = -1)
{
    MiniCam m;
    m.width   = cam.width();
    m.height  = cam.height();
    m.fx = cam.fx();  m.fy = cam.fy();
    m.cx = cam.cx();  m.cy = cam.cy();
    m.tanfovx = (cam.fx() > 1e-6f) ? 0.5f * cam.width()  / cam.fx() : 1.f;
    m.tanfovy = (cam.fy() > 1e-6f) ? 0.5f * cam.height() / cam.fy() : 1.f;
    // m.tanfovx = std::tan(cam.fx() * 0.5f);
    // m.tanfovy = std::tan(cam.fy() * 0.5f);

    m.c2w = c2w.clone();
    m.w2c = torch::linalg_inv(c2w);
    m.frame_id = frame_id;
    return m;
}

} // namespace sv
