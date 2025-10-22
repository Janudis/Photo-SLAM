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
                          const torch::Tensor& c2w, int im_height, int im_width,
                          int frame_id = -1)
{
    MiniCam m;
    // m.width   = cam.width();
    // m.height  = cam.height();
    m.width   = im_width;
    m.height  = im_height;
    m.fx = cam.fx();  m.fy = cam.fy();
    // m.cx = cam.cx();  m.cy = cam.cy();
    m.cx = im_width * 0.5;  m.cy = im_height * 0.5;
    
    // float w_ss = static_cast<float>(im_width) / static_cast<float>(cam.width());
    // float h_ss = static_cast<float>(im_height) / static_cast<float>(cam.height());
    // m.cx = cam.cx() * w_ss;  m.cy = cam.cy() * h_ss;
    // std::cout << "cx,cy= " << m.cx << "," << m.cy << " tested cx, cy= " << im_width * 0.5 << "," << im_height * 0.5 << std::endl; // cx,cy= 79.66075897,63.82849884 tested cx, cy= 80.00000000,60.00000000

    // float fovx = graphics_utils::focal2fov(m.fx, m.width);
    // float fovy = graphics_utils::focal2fov(m.fy, m.height);
    const float fovx = graphics_utils::focal2fov(cam.fx(), cam.width());
    const float fovy = graphics_utils::focal2fov(cam.fy(), cam.height());
    m.tanfovx = std::tan(fovx * 0.5f);
    m.tanfovy = std::tan(fovy * 0.5f);

    m.c2w = c2w.clone();
    m.w2c = torch::linalg_inv(c2w);
    m.frame_id = frame_id;

    // std::cout << "minicam.h: fovx = " << fovx << " fovy = " << fovy << " cx,cy=" << m.cx << "," << m.cy << " im_width,im_height=" << im_width << "," << im_height << std::endl;
    // std::cout << "m.c2w" << m.c2w << std::endl;
    // std::cout << "m.w2c" << m.w2c << std::endl;
    // std::cout << "m.frame_id = " << m.frame_id << std::endl;

    return m;
}
} // namespace sv
