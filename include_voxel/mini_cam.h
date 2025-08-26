#pragma once
#include <torch/torch.h>
#include "include_voxel/voxel_camera.h"
#include "include/graphics_utils.h"

namespace sv {

struct MiniCam {
    torch::Tensor c2w;          
    torch::Tensor w2c;          
    int   width   = 0;
    int   height  = 0;
    float fx      = 0.f;
    float fy      = 0.f;
    float cx      = 0.f;
    float cy      = 0.f;
    float tanfovx = 0.f;        
    float tanfovy = 0.f;      
    float near    = 0.02f;
    std::string cam_mode = "persp";
    int   frame_id = -1;
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

    float fovx = graphics_utils::focal2fov(m.fx, m.width);
    float fovy = graphics_utils::focal2fov(m.fy, m.height);
    m.tanfovx = std::tan(fovx * 0.5f);
    m.tanfovy = std::tan(fovy * 0.5f);

    m.c2w = c2w.clone();
    m.w2c = torch::linalg_inv(c2w);
    m.frame_id = frame_id;

    // std::cout << "minicam.h: fovx = " << fovx << " tanfovx = " << m.tanfovx
    //           << ", cx = " << m.cx
    //           << std::endl;
    // std::cout << "m.c2w" << m.c2w << std::endl;
    // std::cout << "m.w2c" << m.w2c << std::endl;
    // std::cout << "m.frame_id = " << m.frame_id << std::endl;

    return m;
}
} // namespace sv
