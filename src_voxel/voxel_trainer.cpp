    #include "include_voxel/voxel_trainer.h"
    #include "include_voxel/py_utils.h"
    #include "include_voxel/mini_cam.h"
    #include "include_voxel/voxel_constants.h"

    #include <pybind11/embed.h>
    #include <pybind11/numpy.h>
    #include <iostream>
    #include <torch/extension.h>
    #include <limits>  // at the top of the file

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
        oct_level_    = torch::empty({0},     torch::kInt32).to(torch::kCUDA);
        subdiv_meta_  = torch::empty({0},     torch::kFloat32).to(torch::kCUDA);
        subdiv_p_ = torch::empty({0}, torch::kFloat32).to(torch::kCUDA);
    }

    std::unordered_map<std::string, torch::Tensor>
    VoxelTrainer::render(const MiniCam& cam,
                        const py::array_t<uint8_t>& rgb_image,
                        const std::string& output_dir)
    {
        py::gil_scoped_acquire gil;       
        static py::object py_render = py::module_::import(
            "scripts_voxel.python_svraster_bridge.renderer_wrapper")
            .attr("render");

        if (center_.numel() == 0) {
            std::cout << "[INFO] Skipping render — empty voxel data\n";
            return {};
        }
        // === Sanity check before rendering ===
        if (!subdiv_p_.defined() || !subdiv_p_.isfinite().all().item<bool>()) {
            std::cerr << "[ERROR] subdiv_p_ contains NaNs or is not finite.\n";
            std::cerr << " - numel: " << subdiv_p_.numel() << "\n";
            throw std::runtime_error("Invalid subdiv_p_ before rendering.");
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
        dict["octlevels"]    = py::cast(oct_level_.cpu());
        dict["subdiv_meta"]  = py::cast(subdiv_meta_.cpu());    
        dict["subdiv_p"] = py::cast(subdiv_p_.cpu());

        // build Python cam
        py::object py_cam = MiniCam_to_py(cam);
        // call and get back a dict
        py::object out = py_render(py_cam, dict, rgb_image, output_dir);
        torch::Tensor rgb_t = out.attr("get")("rgb").cast<torch::Tensor>();

        return { {"rgb", rgb_t} };
    }

    void VoxelTrainer::set_voxels(torch::Tensor center,
                                torch::Tensor size,
                                torch::Tensor geo,
                                torch::Tensor sh0,
                                torch::Tensor shs,
                                torch::Tensor opacity,
                                torch::Tensor octpath,
                                torch::Tensor octlevel,
                                torch::Tensor subdiv_meta,
                                torch::Tensor subdiv_p)
    {
        center_   = std::move(center) .set_requires_grad(true);   
        size_     = std::move(size)   .set_requires_grad(true);  
        geo_      = std::move(geo)    .set_requires_grad(true);
        sh0_      = std::move(sh0)    .set_requires_grad(true);
        shs_      = std::move(shs)    .set_requires_grad(true);
        opacity_  = std::move(opacity).set_requires_grad(true);
        oct_path_ = std::move(octpath);          // stays fixed
        oct_level_   = std::move(octlevel);              
        subdiv_meta_ = std::move(subdiv_meta).set_requires_grad(true);
        subdiv_meta_.retain_grad();     
        subdiv_p_ = std::move(subdiv_p).set_requires_grad(true);
        subdiv_p_.retain_grad(); 
        subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_).to(subdiv_p_.device());
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
            oct_path_.cpu(),
            oct_level_.cpu(),
            subdiv_meta_.cpu()
        };
        torch::save(pack, p.string());
    }

    std::vector<torch::Tensor> VoxelTrainer::parameters()
    {
            return {
            geo_,
            sh0_,
            shs_,
            opacity_,
            subdiv_meta_
        };
    }

    torch::Tensor VoxelTrainer::get_tensor(const std::string& name) const
    {
        if (name == "subdiv_meta")   return subdiv_meta_;
        else if (name == "subdiv_p") return subdiv_p_;
        else if (name == "oct_level") return oct_level_;
        else if (name == "oct_path")  return oct_path_;
        else if (name == "size")      return size_;        
        throw std::runtime_error(
            "VoxelTrainer::get_tensor(): unknown tensor name \"" + name + "\"");
    }

    void VoxelTrainer::subdivide(const torch::Tensor& mask)
    {
        TORCH_CHECK(mask.dtype() == torch::kBool,
                    "subdivide(mask): mask must be boolean");

        auto device = center_.device();
        torch::NoGradGuard _;

        // parent indices
        torch::Tensor idx_parent = mask.nonzero().view(-1);
        if (idx_parent.numel() == 0) return;

        // ─── INSERT ─── validate that every idx_parent is in [0, N)
        {
            int64_t N = center_.size(0);
            auto max_idx = idx_parent.max().item<int64_t>();
            auto min_idx = idx_parent.min().item<int64_t>();
            if (min_idx < 0 || max_idx >= N) {
                std::cerr << "[ERROR] subdivide(): idx_parent out of bounds. "
                        << "min=" << min_idx << " max=" << max_idx
                        << " but N=" << N << "\n";
                throw std::runtime_error("subdivide(): invalid parent index");
            }
        }

        // torch::Tensor lvl_parent = oct_level_.index({idx_parent});
        // torch::Tensor below_max  = lvl_parent < sv::MAX_OCT_LEVEL;
        torch::Tensor parent_level = oct_level_.index({idx_parent});  // ← rename for clarity
        torch::Tensor below_max    = parent_level < sv::MAX_OCT_LEVEL;
        if (!below_max.any().item<bool>()) return;
        // keep only parents under the level cap
        idx_parent = idx_parent.index({below_max});

        parent_level  = parent_level.index({below_max});

        torch::Tensor idx_keep = (~mask).nonzero().view(-1);

        // ---------- keep the parents that are not subdivided ----------
        auto keep_t       = [&](const torch::Tensor& t){ return t.index({idx_keep}); };

        torch::Tensor center_k  = keep_t(center_);
        torch::Tensor size_k    = keep_t(size_);
        torch::Tensor geo_k     = keep_t(geo_);
        torch::Tensor sh0_k     = keep_t(sh0_);
        torch::Tensor shs_k     = keep_t(shs_);
        torch::Tensor opacity_k = keep_t(opacity_);
        torch::Tensor oct_k     = keep_t(oct_path_);
        torch::Tensor lvl_k     = keep_t(oct_level_);
        torch::Tensor meta_k    = keep_t(subdiv_meta_);

        // ---------- create 8 children for every parent ----------
        int64_t n_parent = idx_parent.size(0);
        int64_t n_child  = n_parent * 8;

        // gather parent attributes
        auto gather = [&](const torch::Tensor& t){ return t.index({idx_parent}); };

        torch::Tensor c_par  = gather(center_);          // (P,3)
        torch::Tensor s_par  = gather(size_);            // (P)
        torch::Tensor geo_par= gather(geo_);
        torch::Tensor sh0_par= gather(sh0_);
        torch::Tensor shs_par= gather(shs_);
        torch::Tensor op_par = gather(opacity_);
        // torch::Tensor path_p = gather(oct_path_).index({below_max});
        torch::Tensor path_p = oct_path_.index({idx_parent});
        // torch::Tensor lvl_p  = gather(oct_level_);
        torch::Tensor lvl_p = oct_level_.index({idx_parent});  // optional

        // new size
        torch::Tensor s_child = (s_par * 0.5).repeat({8});           // (C)
        // don’t create voxels smaller than the minimum allowed size
        s_child = torch::clamp(
        s_child,
        torch::full_like(s_child, sv::MIN_VOX_SIZE),
        torch::full_like(s_child, std::numeric_limits<float>::infinity()));
        // generate 8 offsets (0/1) * size_child
        torch::Tensor offs = torch::tensor({
            { -0.25, -0.25, -0.25},
            {  0.25, -0.25, -0.25},
            { -0.25,  0.25, -0.25},
            {  0.25,  0.25, -0.25},
            { -0.25, -0.25,  0.25},
            {  0.25, -0.25,  0.25},
            { -0.25,  0.25,  0.25},
            {  0.25,  0.25,  0.25},
        }, device).to(torch::kFloat32);                                   // (8,3)

        // broadcast parent centres and add offsets*size
        torch::Tensor c_child = c_par.repeat_interleave(8, /*dim=*/0) +
                                (offs.repeat({n_parent,1}) * s_par.repeat_interleave(8).unsqueeze(1));

        // inherits attributes
        auto repeat8 = [&](const torch::Tensor& t){ return t.repeat_interleave(8, 0); };

        torch::Tensor geo_child  = repeat8(geo_par);
        torch::Tensor sh0_child  = repeat8(sh0_par);
        torch::Tensor shs_child  = repeat8(shs_par);
        torch::Tensor op_child   = repeat8(op_par);

        // levels & morton‑like paths
        // torch::Tensor lvl_child  = lvl_parent.index({below_max}).repeat_interleave(8) + 1;
        torch::Tensor lvl_child = parent_level.repeat_interleave(8) + 1;
        // torch::Tensor idx8 = torch::arange(8, path_p.options()).repeat({n_parent});
        torch::Tensor idx8 = torch::arange(8, path_p.options().dtype(torch::kLong)).repeat({n_parent});
        torch::Tensor path_child = ((path_p * 8).repeat_interleave(8)) | idx8;

        // empty subdiv‑meta for children
        torch::Tensor meta_child = torch::zeros({n_child}, torch::kFloat32).to(device);

        // ---------- concatenate keep & child  ----------
        center_      = torch::cat({center_k,  c_child}).set_requires_grad(true);
        size_        = torch::cat({size_k,    s_child }).set_requires_grad(true);
        geo_         = torch::cat({geo_k,     geo_child}).set_requires_grad(true);
        sh0_         = torch::cat({sh0_k,     sh0_child}).set_requires_grad(true);
        shs_         = torch::cat({shs_k,     shs_child}).set_requires_grad(true);
        opacity_     = torch::cat({opacity_k, op_child}).set_requires_grad(true);
        oct_path_    = torch::cat({oct_k,     path_child});
        oct_level_   = torch::cat({lvl_k,     lvl_child});
        subdiv_meta_ = torch::cat({meta_k, meta_child}).set_requires_grad(true);
        subdiv_meta_.retain_grad();  // ← re-enable grad retention after subdivision
        // Reset subdiv_p_ to zeros of same shape as subdiv_meta_
        subdiv_p_ = torch::zeros_like(subdiv_meta_).set_requires_grad(true);
        subdiv_p_.retain_grad();

        // ——— keep your subdiv-grad buffer in sync ———
        {
            // old buffer and new size
            auto old_buf = subdiv_p_grad_buffer_;
            int64_t oldN  = old_buf.numel();
            int64_t newN  = subdiv_p_.size(0);

            if (oldN != newN) {
                // make a new zeroed buffer of the correct length
                auto new_buf = torch::zeros_like(subdiv_p_);
                // copy over the old entries
                if (oldN > 0) {
                    new_buf.narrow(0, 0, oldN).copy_(old_buf);
                }
                subdiv_p_grad_buffer_ = std::move(new_buf);
            }
        }
    }

    void VoxelTrainer::prune(const torch::Tensor& mask_keep)
    {
        // mask_keep: (N,) bool — stay on *same* device
        // center_   = center_  .index({mask_keep});
        center_ = center_.index({mask_keep}).set_requires_grad(true);
        size_     = size_    .index({mask_keep});
        geo_      = geo_     .index({mask_keep});
        sh0_      = sh0_     .index({mask_keep});
        shs_      = shs_     .index({mask_keep});
        opacity_  = opacity_ .index({mask_keep});
        oct_path_ = oct_path_.index({mask_keep});
        oct_level_= oct_level_.index({mask_keep});
        // subdiv_meta_ = subdiv_meta_.index({mask_keep});
        subdiv_meta_ = subdiv_meta_.index({mask_keep}).set_requires_grad(true);
        subdiv_meta_.retain_grad();  // ← needed again after indexing
        subdiv_p_ = subdiv_p_.index({mask_keep}).set_requires_grad(true);
        subdiv_p_.retain_grad();
        subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);
    }

    void VoxelTrainer::set_subdiv_meta(const torch::Tensor& updated) {
        subdiv_meta_ = updated.clone().set_requires_grad(true);
        subdiv_meta_.retain_grad();
        // subdiv_meta_.grad().detach_();  // ensure old gradient is cleared
        // only clear the old grad if one actually exists
        if (auto g = subdiv_meta_.grad(); g.defined()) {
            g.detach_();
        }
    }

    torch::Tensor VoxelTrainer::get_subdiv_priority_grad() const {
        if (!subdiv_p_.defined() || !subdiv_p_.requires_grad()) {
            throw std::runtime_error("[ERROR] subdiv_p is not properly initialized for gradient.");
        }
        auto grad = subdiv_p_.grad();
        if (!grad.defined()) {
            throw std::runtime_error("[ERROR] subdiv_p.grad() is undefined. Call loss.backward() first.");
        }
        return grad;
    }

    void VoxelTrainer::accumulate_subdiv_gradients(const torch::Tensor& parent_idx, const torch::Tensor& parent_grads) {
        if (!subdiv_p_grad_buffer_.defined()) {
            subdiv_p_grad_buffer_ = torch::zeros_like(subdiv_p_);
        }

        TORCH_CHECK(subdiv_p_grad_buffer_.sizes() == subdiv_p_.sizes(),
                    "subdiv_p_grad_buffer_ and subdiv_p_ size mismatch");

        subdiv_p_grad_buffer_.scatter_add_(
            /*dim=*/0,
            /*index=*/parent_idx,
            /*src=*/parent_grads
        );
    }

    } // namespace sv
