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
    // Misc
    std::string cam_mode = "persp";
    int   frame_id = -1;

    // Derived vectors for SVRaster helpers
    torch::Tensor position;          // [3] float32 CUDA  (c2w[:3, 3])
    torch::Tensor lookat;            // [3] float32 CUDA  (c2w[:3, 2], unit)
    // Pixel footprint used by mark_max_samp_rate (≈ 2*tanfovx / width)
    float pix_size = 0.f;
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

// --- internal helpers ---
inline torch::Tensor ensure_cuda_f32(const torch::Tensor& t) {
    auto out = t;
    if (out.dtype() != torch::kFloat32) out = out.to(torch::kFloat32);
    if (!out.is_cuda()) out = out.to(torch::kCUDA);
    return out.contiguous();
}

inline torch::Tensor mat_inv_cuda_f32(const torch::Tensor& m44) {
    return ensure_cuda_f32(torch::linalg_inv(m44));
}

inline torch::Tensor vec_norm(const torch::Tensor& v) {
    auto n = torch::norm(v);
    // avoid div-by-zero; if zero, just return v
    if (n.item<float>() > 0.f) return v / n;
    return v;
}

/**
 * Build a MiniCam from Photo-SLAM’s Camera + world pose.
 * This REPLACES the old Python MiniCam and your previous fromCamera().
 * - Computes tanfovx/tanfovy from intrinsics.
 * - Fills position = c2w[:3,3], lookat = normalize(c2w[:3,2]).
 * - Sets pix_size = 2*tanfovx/width.
 * - Ensures all tensors are float32 CUDA.
 */
inline MiniCam ToMiniCam(const Camera& cam,
                         const torch::Tensor& c2w_in,
                         int im_height,
                         int im_width,
                         int frame_id = -1)
{
    MiniCam m;
    // Image size (explicit overrides are consistent with your current usage)
    m.width  = im_width;
    m.height = im_height;

    // Intrinsics
    m.fx = cam.fx();
    m.fy = cam.fy();
    // You were forcing principal point to image center; keep that unless you want cam.cx()/cam.cy()
    m.cx = im_width  * 0.5f;
    m.cy = im_height * 0.5f;

    // FOVs → tan(FOV/2)
    const float fovx = graphics_utils::focal2fov(cam.fx(), cam.width());
    const float fovy = graphics_utils::focal2fov(cam.fy(), cam.height());
    m.tanfovx = std::tan(fovx * 0.5f);
    m.tanfovy = std::tan(fovy * 0.5f);

    // Poses: ensure CUDA float32 & contiguous
    m.c2w = ensure_cuda_f32(c2w_in);
    m.w2c = mat_inv_cuda_f32(m.c2w);

    // Derived vectors
    m.position = ensure_cuda_f32(m.c2w.index({torch::indexing::Slice(0,3), 3}));
    auto forward = ensure_cuda_f32(m.c2w.index({torch::indexing::Slice(0,3), 2}));
    m.lookat = vec_norm(forward);  // keep unit; python version didn’t normalize, but safe to normalize here

    // Pixel footprint used by sampling-rate helper
    m.pix_size = (m.width > 0) ? (2.f * m.tanfovx / static_cast<float>(m.width)) : 0.f;

    m.cam_mode = "persp";
    m.frame_id = frame_id;
    // near already set to 0.02f to match SVRaster defaults

    return m;
}

} // namespace sv
