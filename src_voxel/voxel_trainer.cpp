#include "include_voxel/voxel_trainer.h"
#include "include_voxel/py_utils.h"
#include "include_voxel/mini_cam.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <torch/extension.h>

namespace py = pybind11;
namespace sv {

VoxelTrainer::VoxelTrainer(int grid_res)
: G_(grid_res)
{
    // initialize as empty on CUDA
    center_   = torch::empty({0,3},   torch::kFloat32).to(torch::kCUDA);
    size_     = torch::empty({0},     torch::kFloat32).to(torch::kCUDA);
    geo_      = torch::empty({0,8},   torch::kFloat32).to(torch::kCUDA);
    sh0_      = torch::empty({0,3},   torch::kFloat32).to(torch::kCUDA);
    shs_      = torch::empty({0,45},  torch::kFloat32).to(torch::kCUDA);
    opacity_ = torch::empty({0}, torch::kFloat32).to(torch::kCUDA);
    oct_path_ = torch::empty({0},     torch::kLong).to(torch::kCUDA);
}

std::unordered_map<std::string, torch::Tensor>
VoxelTrainer::render(const MiniCam& cam,
                     const py::array_t<uint8_t>& rgb_image,
                     const std::string& output_dir)
{
    // std::fprintf(stderr, "[DBG]  reached  render()  (thread %lu)\n",
    //              pthread_self());
    py::gil_scoped_acquire gil;          // <‑‑ keep this here
    // std::fprintf(stderr, "[DBG]  GIL acquired (thread %lu)\n",
    //              pthread_self());
    static py::object py_render = py::module_::import(
        "scripts_voxel.python_svraster_bridge.renderer_wrapper")
        .attr("render");
    // std::fprintf(stderr, "[DBG]  Python module imported\n");    
    if (center_.numel() == 0) {
        std::cout << "[INFO] Skipping render — empty voxel data\n";
        return {};
    }

    // pack into Python dict
    py::dict dict;
    dict["octpaths"]    = py::cast(oct_path_.cpu());
    dict["centers"]     = py::cast(center_.cpu());
    dict["vox_lengths"] = py::cast(size_.cpu());
    // only first 6 dims of geo
    dict["cov3D"]       = py::cast(geo_.slice(1,0,6).contiguous().cpu());
    dict["colors"]      = py::cast(sh0_.cpu());
    dict["shs"]         = py::cast(shs_.cpu());
    dict["opacities"]   = py::cast(opacity_.cpu());  //# (N,)

    // build Python cam
    py::object py_cam = MiniCam_to_py(cam);

    // call and get back a dict
    py::object out = py_render(py_cam, dict, rgb_image, output_dir);

    // // extract the "rgb" entry
    // py::array_t<float> rgb_np = out.attr("get")("rgb").cast<py::array_t<float>>();
    // auto buf = rgb_np.request();
    // auto H = cam.height, W = cam.width;

    // torch::Tensor rgb_t = torch::from_blob(
    //     buf.ptr, {3, H, W}, torch::kFloat32).clone();
    torch::Tensor rgb_t = out.attr("get")("rgb").cast<torch::Tensor>();
    
    return { {"rgb", rgb_t} };
}

void VoxelTrainer::set_voxels(torch::Tensor center,
                              torch::Tensor size,
                              torch::Tensor geo,
                              torch::Tensor sh0,
                              torch::Tensor shs,
                              torch::Tensor opacity,
                              torch::Tensor octpath)
{
    // center_   = std::move(center);
    // size_     = std::move(size);
    // geo_      = std::move(geo).set_requires_grad(true);
    // sh0_      = std::move(sh0).set_requires_grad(true);
    // shs_      = std::move(shs).set_requires_grad(true);
    // opacity_  = std::move(opacity).set_requires_grad(true);
    // oct_path_ = std::move(octpath);
    center_   = std::move(center) .set_requires_grad(true);   
    size_     = std::move(size)   .set_requires_grad(true);  
    geo_      = std::move(geo)    .set_requires_grad(true);
    sh0_      = std::move(sh0)    .set_requires_grad(true);
    shs_      = std::move(shs)    .set_requires_grad(true);
    opacity_  = std::move(opacity).set_requires_grad(true);
    oct_path_ = std::move(octpath);          // stays fixed
}

void VoxelTrainer::save_torch(const std::filesystem::path& p) const
{
    // save all tensors on CPU
    auto pack = std::vector<torch::Tensor>{
        center_.cpu(),
        size_.cpu(),
        geo_.cpu(),
        sh0_.cpu(),
        shs_.cpu(),
        opacity_.cpu(),
        oct_path_.cpu()
    };
    torch::save(pack, p.string());
}

std::vector<torch::Tensor> VoxelTrainer::parameters()
{
    // return { center_, size_, geo_, sh0_, shs_, opacity_, oct_path_ };
        return {
        geo_,
        sh0_,
        shs_,
        opacity_,
    };
}

} // namespace sv
