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
    float near    = 0.01f;
    std::string cam_mode = "persp";
    int   frame_id = -1;
    torch::Tensor position;   // [3] CPU
    torch::Tensor lookat;     // [3] CPU
    float pix_size = 0.f;
};

/* -------- helper: build MiniCam from a C++ Camera + pose ---------------- */
inline MiniCam fromCamera(const Camera& cam,
                          const torch::Tensor& c2w, int im_height, int im_width,
                          int frame_id = -1)
{
    MiniCam m;
    m.width   = im_width;
    m.height  = im_height;

    // Pyramid-aware intrinsics:
    // scale native camera intrinsics to the render/training resolution.
    // This keeps consistency for both full-res and Gaussian-Pyramid levels.
    const int native_w = (cam.width()  > 0) ? cam.width()  : im_width;
    const int native_h = (cam.height() > 0) ? cam.height() : im_height;
    const float w_ss = static_cast<float>(im_width)  / static_cast<float>(native_w);
    const float h_ss = static_cast<float>(im_height) / static_cast<float>(native_h);

    m.fx = cam.fx() * w_ss;
    m.fy = cam.fy() * h_ss;
    m.cx = cam.cx() * w_ss;
    m.cy = cam.cy() * h_ss;

    const float fovx = graphics_utils::focal2fov(m.fx, m.width);
    const float fovy = graphics_utils::focal2fov(m.fy, m.height);
    m.tanfovx = std::tan(fovx * 0.5f);
    m.tanfovy = std::tan(fovy * 0.5f);

    m.c2w = c2w.clone();
    m.w2c = torch::linalg_inv(c2w);
    m.frame_id = frame_id;

    m.position = m.c2w.index({torch::indexing::Slice(0,3), 3}).clone();  // [3] CPU
    m.lookat   = m.c2w.index({torch::indexing::Slice(0,3), 2}).clone();  // [3] CPU
    // normalize lookat to be safe
    auto norm = m.lookat.norm().item<float>();
    if (norm > 1e-6f) {
        // std::cout << "minicam.h: normalizing lookat vector, norm = " << norm << std::endl;
        m.lookat = m.lookat / norm;
    }
    m.pix_size = 2.f * m.tanfovx / m.width;
    // std::cout << "minicam.h: fovx = " << fovx << " fovy = " << fovy << " cx,cy=" << m.cx << "," << m.cy << " im_width,im_height=" << im_width << "," << im_height << std::endl;
    // std::cout << "m.c2w" << m.c2w << std::endl;
    // std::cout << "m.w2c" << m.w2c << std::endl;
    // std::cout << "m.frame_id = " << m.frame_id << std::endl;

    return m;
}
} // namespace sv
