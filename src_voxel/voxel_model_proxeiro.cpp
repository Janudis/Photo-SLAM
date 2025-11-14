void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd, const std::vector<sv::MiniCam>& cams)
{
    namespace py = pybind11;
    std::cout << "VoxelModel::createFromPcd() called with " << pcd.size() << " points.\n";

    TORCH_CHECK(global_scene_extent_ > 0.f, "global_scene_extent_ must be set (>0).");
    TORCH_CHECK(fixed_vox_size_   > 0.f, "fixed_vox_size_ must be set (>0).");
    TORCH_CHECK(max_sh_degree_ >= 0, "max_sh_degree_ must be >= 0.");

    const int N = static_cast<int>(pcd.size());
    if (N == 0) { std::cerr << "[createFromPcd] Empty PCD — nothing to do.\n"; return; }

    auto dev = torch::kCUDA;

    // --- 0) Pack PCD to CPU tensors (xyz in world, rgb in [0..1]) -------------------
    torch::Tensor xyz_cpu = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor rgb_cpu = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(torch::kCPU));
    {
        int i = 0;
        for (const auto& kv : pcd) {
            const auto& P = kv.second;
            xyz_cpu[i][0] = P.xyz_(0);
            xyz_cpu[i][1] = P.xyz_(1);
            xyz_cpu[i][2] = P.xyz_(2);
            // P.color_ already scaled? If it’s 0..255, divide; if already 0..1, keep.
            rgb_cpu[i][0] = P.color_(0);  // adapt if needed
            rgb_cpu[i][1] = P.color_(1);
            rgb_cpu[i][2] = P.color_(2);
            ++i;
        }
    }

    // --- 1) Estimate a compact main-scene bbox from the PCD --------------------------
    py::gil_scoped_acquire gil;
    // Keep this order identical to the original working version:
    static py::module svr_utils = py::module::import("svraster_cuda").attr("utils");
    // Insert path *once* via the svm_mod lambda, then any src.* imports are safe.
    static py::module svm_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.sparse_voxel_model");
    }();
    // Now it's safe to import from src.utils.*
    static py::module act_mod     = py::module::import("src.utils.activation_utils");
    static py::module oct_utils   = py::module::import("src.utils.octree_utils");
    static py::module bound_utils = py::module::import("src.utils.bounding_utils"); // only if you use it
    static py::module types       = py::module::import("types");
    static py::module torch_mod   = py::module::import("torch");

    py::object xyz_cpu_py = py::cast(xyz_cpu.contiguous());
    py::object ns         = types.attr("SimpleNamespace")("points"_a = xyz_cpu_py.attr("numpy")());
    // tweak 0.1 → your density fraction / percentile as you like
    py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(ns, py::float_(0.1f));
    // back to tensors, CUDA
    py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]);
    auto center_t = center_t_py.cast<torch::Tensor>().to(dev).contiguous();        // [3]
    float radius  = cr[1].cast<float>();
    auto radius_t = torch::full({3}, radius, torch::dtype(torch::kFloat32).device(dev));

    scene_center_ = torch::tensor(
        {global_scene_center_[0], global_scene_center_[1], global_scene_center_[2]},
        torch::dtype(torch::kFloat32).device(dev)).contiguous();                                             
    int   outside_level = 0;
    const float scene_extent_scalar = global_scene_extent_ * std::pow(2.0f, outside_level);
    scene_extent_ = torch::tensor({scene_extent_scalar},
        torch::dtype(torch::kFloat32).device(dev)).contiguous();
    inside_extent_ = torch::tensor({global_scene_extent_},
        torch::dtype(torch::kFloat32).device(dev)
    ).contiguous();    
    scene_min_t_  = (scene_center_ - 0.5f * scene_extent_).contiguous(); // [3]

    // --- 2) Fix base level from fixed_vox_size_ and cache effective voxel size ------
    const int MAX_L = max_num_levels_;
    auto vox_size_t = torch::full({1}, fixed_vox_size_, torch::dtype(torch::kFloat32).device(dev));
    py::object L_fp_py = oct_utils.attr("vox_size_2_level")(py::cast(scene_extent_), py::cast(vox_size_t));
    auto L_fp      = L_fp_py.cast<torch::Tensor>().round();
    auto L_clamped = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8).contiguous();  // [1]
    octlevel_      = L_clamped.item<int8_t>();

    py::object vox_eff_py = oct_utils.attr("level_2_vox_size")(
        py::cast(scene_extent_), py::cast(L_clamped.view({1,1})));
    vox_eff_ = vox_eff_py.cast<torch::Tensor>().view({1,1}).contiguous();               // [1,1]

    // --- 3) Convert bbox → dense ijk range at base level -----------------------------
    auto bb_min = (scene_center_ - radius_t).contiguous();
    auto bb_max = (scene_center_ + radius_t).contiguous();

    auto vox_t = vox_eff_.mean().view({1}).repeat({3}).contiguous(); // [3] CUDA float
    const int64_t grid_limit = (1LL << static_cast<int>(octlevel_));
    auto ijk_min = ((bb_min - scene_min_t_) / vox_t).floor().to(torch::kLong);
    auto ijk_max = ((bb_max - scene_min_t_) / vox_t).ceil().to(torch::kLong) - 1;

    // Clamp
    auto zero = torch::zeros_like(ijk_min);
    auto lim  = torch::full_like(ijk_max, grid_limit - 1);
    ijk_min = torch::maximum(ijk_min, zero);
    ijk_max = torch::minimum(ijk_max, lim);

    if (!(ijk_min <= ijk_max).all().item<bool>()) {
        std::cerr << "[createFromPcd] Degenerate bbox → no cells.\n";
        return;
    }

    // Enumerate ijk box
    auto ir = torch::arange(ijk_min[0].item<int64_t>(), ijk_max[0].item<int64_t>() + 1,
                            torch::dtype(torch::kLong).device(dev));
    auto jr = torch::arange(ijk_min[1].item<int64_t>(), ijk_max[1].item<int64_t>() + 1,
                            torch::dtype(torch::kLong).device(dev));
    auto kr = torch::arange(ijk_min[2].item<int64_t>(), ijk_max[2].item<int64_t>() + 1,
                            torch::dtype(torch::kLong).device(dev));
    auto grids  = torch::meshgrid({ir, jr, kr}, /*indexing=*/"ij");
    auto ijk_box= torch::stack(
        { grids[0].contiguous().view({-1}),
          grids[1].contiguous().view({-1}),
          grids[2].contiguous().view({-1}) }, 1).contiguous(); // [Nc,3]
    int64_t Nc = ijk_box.size(0);
    auto L_box = torch::full({Nc,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous();

    // Morton/octpath for those cells
    py::object octpath_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_box), py::cast(L_box));
    auto octpath = octpath_py.cast<torch::Tensor>().contiguous(); // [Nc,1] int64
    
    if (!cams.empty())
    {
        py::gil_scoped_acquire gil;
        static py::module oct_utils = py::module::import("src.utils.octree_utils");
        static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");

        // Decode voxel centers/sizes for filtering
        py::tuple dec = oct_utils.attr("octpath_decoding")(
            py::cast(octpath.contiguous()),
            py::cast(L_box.contiguous()),
            py::cast(scene_center_.contiguous()),
            py::cast(scene_extent_.contiguous())
        );
        at::Tensor vox_center = dec[0].cast<at::Tensor>(); // [Nu,3]
        at::Tensor vox_size   = dec[1].cast<at::Tensor>(); // [Nu,1]
        std::cout << "octpath dev="   << (octpath.is_cuda() ? "cuda" : "cpu") 
            << " dtype="        << octpath.dtype() << "\n";
        std::cout << "vox_center dev="<< (vox_center.is_cuda() ? "cuda" : "cpu")
                << " dtype="        << vox_center.dtype() << "\n";
        std::cout << "vox_size dev="  << (vox_size.is_cuda() ? "cuda" : "cpu")
                << " dtype="        << vox_size.dtype() << "\n";

        // Build Python list of MiniCams (ensure CUDA tensors)
        py::list py_cams;
        py::module torch_mod = py::module::import("torch");
        py::object py_cuda = torch_mod.attr("device")("cuda");
        auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
            if (py::hasattr(obj, name)) {
                py::object t = obj.attr(name);
                // Only tensors have .is_cuda/.to; defensive check:
                if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                    obj.attr(name) = t.attr("to")(py_cuda);  // or t.attr("to")("cuda")
                }
            }
        };
        for (const auto& c : cams) {
            py::object py_cam = MiniCam_to_py(c);
            // matrices
            move_attr_to_cuda_if_tensor(py_cam, "w2c");
            move_attr_to_cuda_if_tensor(py_cam, "c2w");
            // vectors used by mark_near / other helpers
            move_attr_to_cuda_if_tensor(py_cam, "position");
            move_attr_to_cuda_if_tensor(py_cam, "lookat");
            py_cams.append(py_cam);
        }

        auto Nu_before = octpath.size(0);
        // mark_max_samp_rate -> keep rate>0
        at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
            py_cams,
            py::cast(octpath),
            py::cast(vox_center),
            py::cast(vox_size)
        ).cast<at::Tensor>();

        at::Tensor kept = rate > 0;
        int64_t n_rate_pos = kept.sum().item<int64_t>();
        // optional near filtering
        const float near_thresh = 0.2f;
        int64_t n_near_hit = 0;
        if (near_thresh > 0.0f) {
            at::Tensor is_near = svr_mod.attr("mark_near")(
                py_cams,
                py::cast(octpath),
                py::cast(vox_center),
                py::cast(vox_size),
                py::float_(near_thresh)
            ).cast<at::Tensor>();
            kept = kept & (~is_near);
            n_near_hit = is_near.sum().item<int64_t>();
        }

        auto idx = torch::nonzero(kept).view({-1});
        int64_t K = idx.size(0);
        // Apply mask (all tensors together, no reshapes)
        if (K > 0 && K < octpath.size(0)) {
            octpath = octpath.index_select(0, idx).contiguous();   // [K,1]
            L_box   = L_box.index_select(0, idx).contiguous();     // [K,1] already
        }
        // Recompute sizes and assert consistency
        Nc = octpath.size(0);
        // Debug prints
        std::cout << "[filter] Nu_before=" << Nu_before
                << " rate>0=" << n_rate_pos
                << " near_hit=" << n_near_hit
                << " kept_final=" << Nc << std::endl;
    }

    // --- 4) Create or reuse SVM; set scene & topology -------------------------------
    if (!py_->svm || py_->svm.is_none()) {
        py::object SVM = svm_mod.attr("SparseVoxelModel");
        py_->svm = SVM(py::arg("sh_degree") = max_sh_degree_);
    }
    py_->svm.attr("scene_center")  = scene_center_.contiguous();
    py_->svm.attr("scene_extent")  = scene_extent_.contiguous();
    py_->svm.attr("inside_extent") = inside_extent_.contiguous();
    py_->svm.attr("octpath")       = octpath;
    py_->svm.attr("octlevel")      = L_box;

    // --- 5) Initialize learnables ----------------------------------------------------
    // Subdivision priority
    auto subdiv_p = torch::ones({Nc,1}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();
    py_->svm.attr("_subdiv_p") = subdiv_p.detach().requires_grad_();

    // SH-0: use mean RGB of the input PCD as a simple prior (or a constant)
    auto rgb_mean = rgb_cpu.to(dev).mean(0, /*keepdim=*/false).contiguous(); // [3]
    py::object sh0_dc_py = act_mod.attr("rgb2shzero")(py::cast(rgb_mean.view({1,3})));
    auto sh0_dc = sh0_dc_py.cast<torch::Tensor>().expand({Nc,3}).contiguous();
    py_->svm.attr("_sh0") = sh0_dc.detach().requires_grad_();

    // Higher-degree SH zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs = torch::zeros({Nc, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_shs") = shs.detach().requires_grad_();

    // Geometry: grid density (-10) over all grid points
    torch::Tensor grid_pts_key = py_->svm.attr("grid_pts_key").cast<torch::Tensor>(); // [Mg,3] int64
    auto geo_grid = torch::full({grid_pts_key.size(0), 1}, -10.0f,
                                torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_geo_grid_pts") = geo_grid.detach().requires_grad_();

    // Activate initial SH degree (e.g., min(3, max_sh_degree_))
    py_->svm.attr("active_sh_degree") = py::int_(std::min(max_sh_degree_, 3));

    // --- 6) Sync back to C++ members & optimizer groups -----------------------------
    auto fetch = [&](const char* name){ return py_->svm.attr(name).cast<torch::Tensor>().contiguous(); };
    this->oct_path_      = fetch("octpath");
    this->oct_level_     = fetch("octlevel");
    this->center_        = fetch("vox_center");
    this->size_          = fetch("vox_size").squeeze(1);
    this->vox_size_inv_  = 1.0f / size_;
    this->grid_pts_key_  = fetch("grid_pts_key");
    this->vox_key_       = fetch("vox_key");

    this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    this->sh0_           = fetch("_sh0").requires_grad_(true);
    this->shs_           = fetch("_shs").requires_grad_(true);
    this->subdiv_p_      = fetch("_subdiv_p").requires_grad_(true);

    // stats buffer
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));

    // Register with optimizer
    VOXEL_MODEL_TENSORS_TO_VEC

    std::cout << "[createFromPcd] Seeded " << Nc << " voxels from bbox @ level "
              << static_cast<int>(octlevel_) << " (vox_size="
              << vox_eff_.item<float>() << " m)\n";
}

void VoxelModel::increasePcd(std::vector<float> pcd_full,
                             std::vector<float> colors,
                             const int /*iteration*/, const std::vector<sv::MiniCam>& cams)
{
    namespace py = pybind11;

    const int Nf = static_cast<int>(pcd_full.size());
    if (Nf < 3 || colors.size() < 3) return;
    const int N = Nf / 3;

    std::cout << "VoxelModel::increasePcd() called with " << N << " points.\n";
    TORCH_CHECK(global_scene_extent_ > 0.f && fixed_vox_size_ > 0.f,
                "increasePcd: scene extent / fixed vox size not set.");
    TORCH_CHECK(py_->svm && !py_->svm.is_none(),
                "increasePcd: SVM not initialized; call createFromPcd first.");

    // --- 0) Pack batch to CPU tensors (xyz in world, rgb in [0..1 or your scale]) ---
    torch::Tensor xyz_cpu = torch::from_blob(
        pcd_full.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();

    torch::Tensor rgb_cpu = torch::from_blob(
        colors.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone(); // .div_(255.0f);  // uncomment if input colors are 0..255

    auto dev = torch::kCUDA;

    // --- 1) Get a compact bbox via the Python heuristic (median + density) -----------
    py::gil_scoped_acquire gil;
    static py::module bound_utils = py::module::import("src.utils.bounding_utils");
    static py::module types       = py::module::import("types");
    static py::module torch_mod   = py::module::import("torch");
    static py::module svr_utils   = py::module::import("svraster_cuda").attr("utils");
    static py::module act_mod     = py::module::import("src.utils.activation_utils");

    py::object xyz_cpu_py = py::cast(xyz_cpu.contiguous());
    py::object ns         = types.attr("SimpleNamespace")("points"_a = xyz_cpu_py.attr("numpy")());

    // tune 0.1 to your needs; keep consistent with createFromPcd()
    py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(ns, py::float_(0.1f));

    py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]);
    auto center_t = center_t_py.cast<torch::Tensor>().to(dev).contiguous();  // [3]
    float radius  = cr[1].cast<float>();
    auto radius_t = torch::full({3}, radius, torch::dtype(torch::kFloat32).device(dev));

    // --- 2) Convert bbox → dense ijk block at cached base level ----------------------
    // Assumes createFromPcd() has set scene_min_t_ and vox_eff_ (base level size)
    auto bb_min = (center_t - radius_t).contiguous();
    auto bb_max = (center_t + radius_t).contiguous();

    auto vox_t = vox_eff_.mean().view({1}).repeat({3}).contiguous(); // [3] CUDA float
    const int64_t grid_limit = (1LL << static_cast<int>(octlevel_));

    auto ijk_min = ((bb_min - scene_min_t_) / vox_t).floor().to(torch::kLong);
    auto ijk_max = ((bb_max - scene_min_t_) / vox_t).ceil().to(torch::kLong) - 1;

    // Clamp to valid grid range
    auto zero = torch::zeros_like(ijk_min);
    auto lim  = torch::full_like(ijk_max, grid_limit - 1);
    ijk_min = torch::maximum(ijk_min, zero);
    ijk_max = torch::minimum(ijk_max, lim);

    if (!(ijk_min <= ijk_max).all().item<bool>()) {
        std::cout << "[increasePcd] Heuristic bbox produced no in-bounds cells; skipping.\n";
        return;
    }

    // Enumerate the ijk block
    auto ir = torch::arange(ijk_min[0].item<int64_t>(), ijk_max[0].item<int64_t>() + 1,
                            torch::dtype(torch::kLong).device(dev));
    auto jr = torch::arange(ijk_min[1].item<int64_t>(), ijk_max[1].item<int64_t>() + 1,
                            torch::dtype(torch::kLong).device(dev));
    auto kr = torch::arange(ijk_min[2].item<int64_t>(), ijk_max[2].item<int64_t>() + 1,
                            torch::dtype(torch::kLong).device(dev));
    auto grids   = torch::meshgrid({ir, jr, kr}, /*indexing=*/"ij");
    auto ijk_box = torch::stack(
        { grids[0].contiguous().view({-1}),
          grids[1].contiguous().view({-1}),
          grids[2].contiguous().view({-1}) }, 1).contiguous(); // [Nc,3]
    int64_t Nc = ijk_box.size(0);

    auto L_box = torch::full({Nc,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous();

    // Build morton/octpath for candidates
    py::object octpath_box_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_box), py::cast(L_box));
    auto octpath_box = octpath_box_py.cast<torch::Tensor>().contiguous(); // [Nc,1] int64

    // ---- (NEW) SVR-style filtering with cameras ----
    if (!cams.empty()) {
        py::gil_scoped_acquire gil;
        static py::module oct_utils = py::module::import("src.utils.octree_utils");
        static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");
        static py::module torch_mod = py::module::import("torch");

        // Decode voxel centers/sizes
        py::tuple dec = oct_utils.attr("octpath_decoding")(
            py::cast(octpath_box),
            py::cast(L_box),
            py::cast(scene_center_.contiguous()),
            py::cast(scene_extent_.contiguous())
        );
        at::Tensor vox_center = dec[0].cast<at::Tensor>(); // [Nu,3] cuda
        at::Tensor vox_size   = dec[1].cast<at::Tensor>(); // [Nu,1] cuda

        // Build Python list of CUDA MiniCams
        py::list py_cams;
        py::object py_cuda = torch_mod.attr("device")("cuda");
        auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
            if (py::hasattr(obj, name)) {
                py::object t = obj.attr(name);
                if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                    obj.attr(name) = t.attr("to")(py_cuda);
                }
            }
        };
        for (const auto& c : cams) {
            py::object py_cam = MiniCam_to_py(c);
            move_attr_to_cuda_if_tensor(py_cam, "w2c");
            move_attr_to_cuda_if_tensor(py_cam, "c2w");
            move_attr_to_cuda_if_tensor(py_cam, "position");
            move_attr_to_cuda_if_tensor(py_cam, "lookat");
            py_cams.append(py_cam);
        }

        auto Nu_before = octpath_box.size(0);

        // rate > 0
        at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
            py_cams, py::cast(octpath_box), py::cast(vox_center), py::cast(vox_size)
        ).cast<at::Tensor>();
        at::Tensor kept = rate > 0;
        int64_t n_rate_pos = kept.sum().item<int64_t>();

        // near filtering (same threshold you used in createFromPcd)
        const float near_thresh = 0.2f;
        int64_t n_near_hit = 0;
        if (near_thresh > 0.0f) {
            at::Tensor is_near = svr_mod.attr("mark_near")(
                py_cams, py::cast(octpath_box), py::cast(vox_center), py::cast(vox_size),
                py::float_(near_thresh)
            ).cast<at::Tensor>();
            kept = kept & (~is_near);
            n_near_hit = is_near.sum().item<int64_t>();
        }

        auto idx = torch::nonzero(kept).view({-1});
        int64_t K = idx.size(0);

        if (K == 0) {
            std::cout << "[increasePcd/filter] all candidates filtered out, nothing to add.\n";
            return;
        }

        if (K < octpath_box.size(0)) {
            // Apply mask to ALL aligned tensors
            octpath_box = octpath_box.index_select(0, idx).contiguous(); // [K,1]
            L_box         = L_box.index_select(0, idx).contiguous();         // [K,1]
        }

        Nc = octpath_box.size(0); // update Nu after filtering

        // Sanity
        TORCH_CHECK(L_box.sizes() == torch::IntArrayRef({Nc,1}), "L_box shape mismatch after filtering");

        std::cout << "[increasePcd/filter] Nu_before=" << Nu_before
                << " rate>0=" << n_rate_pos
                << " near_hit=" << n_near_hit
                << " kept_final=" << Nc << std::endl;
    }
    // ---- end filtering ----

    // --- 3) Dedup against existing topology + avoid base-level parents if children exist
    auto octpath_cur  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous(); // [No,1]
    auto octlevel_cur = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous(); // [No,1]

    auto key_box = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                 + L_box.view({-1}).to(torch::kInt64);

    torch::Tensor new_mask;
    if (octpath_cur.numel() == 0) {
        new_mask = torch::ones({Nc}, torch::dtype(torch::kBool).device(dev));
    } else {
        static py::module torch_mod_local = py::module::import("torch");
        auto key_cur = octpath_cur.view({-1}).to(torch::kInt64).mul(256)
                     + octlevel_cur.view({-1}).to(torch::kInt64);
        py::object isin_py = torch_mod_local.attr("isin")(py::cast(key_box), py::cast(key_cur));
        auto is_dup = isin_py.cast<torch::Tensor>().to(torch::kBool); // [Nc]
        new_mask = ~is_dup;

        // Drop base-level parents of already-subdivided regions (L_old > base_L)
        const int MAX_L  = max_num_levels_;
        const int base_L = static_cast<int>(octlevel_);
        if (octpath_cur.numel() > 0) {
            auto Lold_i64     = octlevel_cur.view({-1}).to(torch::kInt64);
            auto has_children = (Lold_i64 > base_L);
            if (has_children.any().item<bool>()) {
                // Clear octant bits below base_L and build parent keys at base_L
                const int levels_below  = std::max(0, MAX_L - base_L);
                const int bits_to_clear = 3 * levels_below;
                long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                long long keep_mask_ll  = ~lower_mask;
                auto keep_mask = torch::full({1}, static_cast<int64_t>(keep_mask_ll),
                    torch::TensorOptions().dtype(torch::kInt64).device(dev));

                auto op_old_i64  = octpath_cur.view({-1}).to(torch::kInt64);
                auto op_anc_base = (op_old_i64 & keep_mask);

                auto sel_child = torch::nonzero(has_children).view({-1});
                op_anc_base    = op_anc_base.index_select(0, sel_child);

                auto key_children_as_parent = op_anc_base.mul(256)
                                           + torch::full_like(op_anc_base, static_cast<int64_t>(base_L));

                // unique + sorted helper
                auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
                    TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
                    if (t.numel() <= 1) return t.contiguous();
                    auto sort_res = t.sort(0);
                    auto sorted   = std::get<0>(sort_res);
                    using torch::indexing::Slice;
                    auto keep = torch::empty_like(sorted, torch::kBool);
                    keep.index_put_({0}, true);
                    auto neq = sorted.index({Slice(1, torch::indexing::None)})
                             != sorted.index({Slice(torch::indexing::None, -1)});
                    keep.index_put_({Slice(1, torch::indexing::None)}, neq);
                    auto idx = torch::nonzero(keep).view({-1});
                    return sorted.index_select(0, idx).contiguous();
                };
                key_children_as_parent = unique_sorted_1d(key_children_as_parent);

                py::object isin_py2 = torch_mod_local.attr("isin")(
                    py::cast(key_box), py::cast(key_children_as_parent));
                auto would_collide_parent = isin_py2.cast<torch::Tensor>().to(torch::kBool); // [Nc]
                new_mask = new_mask & (~would_collide_parent);
            }
        }
    }

    if (!new_mask.any().item<bool>()) {
        std::cout << "[increasePcd] No new voxels from bbox.\n";
        return;
    }

    // Select additions
    auto sel        = torch::nonzero(new_mask).view({-1}); // [Nk]
    auto octpath_add= octpath_box.index_select(0, sel);    // [Nk,1]
    auto L_add      = L_box.index_select(0, sel);          // [Nk,1]
    const int Nk    = sel.size(0);

    // --- 4) Append topology ----------------------------------------------------------
    py_->svm.attr("octpath")  = torch::cat({octpath_cur,  octpath_add}, 0).contiguous();
    py_->svm.attr("octlevel") = torch::cat({octlevel_cur, L_add},       0).contiguous();

    // --- 5) Append learnables for new rows -------------------------------------------
    // _subdiv_p
    auto subdiv_old = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
    auto subdiv_add = torch::ones({Nk,1}, torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old, subdiv_add}, 0)
                                 .contiguous().detach().requires_grad_();

    // _sh0: simple prior from this batch's mean color (or constant)
    auto rgb_mean = rgb_cpu.to(dev).mean(0, /*keepdim=*/false).contiguous(); // [3]
    py::object sh0_add_py = act_mod.attr("rgb2shzero")(py::cast(rgb_mean.view({1,3})));
    auto sh0_add_bcast = sh0_add_py.cast<torch::Tensor>().expand({Nk,3}).contiguous();
    auto sh0_old = py_->svm.attr("_sh0").cast<torch::Tensor>();
    py_->svm.attr("_sh0") = torch::cat({sh0_old, sh0_add_bcast}, 0)
                            .contiguous().detach().requires_grad_();

    // _shs: zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs_old = py_->svm.attr("_shs").cast<torch::Tensor>();
    auto shs_add = torch::zeros({Nk, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_shs") = torch::cat({shs_old, shs_add}, 0)
                            .contiguous().detach().requires_grad_();

    // --- 6) Rebuild grid links; grow _geo_grid_pts only if grid expanded -------------
    auto grid_pts_key_new = py_->svm.attr("grid_pts_key").cast<torch::Tensor>(); // [M,3]
    const int64_t M_prev  = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
    const int64_t M_curr  = grid_pts_key_new.size(0);

    if (M_curr > M_prev) {
        auto grow = torch::full({M_curr - M_prev, 1}, -10.0f,
                        torch::dtype(torch::kFloat32).device(dev))
                    .contiguous().detach().requires_grad_();
        appendGroup_(/*group_idx=*/0, grow, "_geo_grid_pts", &this->_geo_grid_pts_);
    }
    // Append SH params to optimizer groups (topology done above)
    appendGroup_(/*group_idx=*/1, sh0_add_bcast, "_sh0", &this->sh0_);
    appendGroup_(/*group_idx=*/2, shs_add,       "_shs", &this->shs_);

    // --- 7) Sync mirrors to C++ for renderer ----------------------------------------
    auto fetch = [&](const char* name){ return py_->svm.attr(name).cast<torch::Tensor>().contiguous(); };
    this->oct_path_      = fetch("octpath");
    this->oct_level_     = fetch("octlevel");
    this->center_        = fetch("vox_center");
    this->size_          = fetch("vox_size").squeeze(1);
    this->vox_size_inv_  = 1.0f / size_;
    this->grid_pts_key_  = fetch("grid_pts_key");
    this->vox_key_       = fetch("vox_key");
    this->subdiv_p_      = py_->svm.attr("_subdiv_p").cast<torch::Tensor>().requires_grad_(true);

    // stats buffer resize
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));

    // --- 8) Re-register with optimizer ----------------------------------------------
    VOXEL_MODEL_TENSORS_TO_VEC

    std::cout << "[increasePcd] Added " << Nk
              << " bbox voxels at level " << static_cast<int>(octlevel_) << ". "
              << "Total now: " << this->oct_path_.size(0) << "\n";
}

void VoxelModel::createFromPcd(const std::map<point3D_id_t, Point3D>& pcd, const std::vector<sv::MiniCam>& cams)
{
    namespace py = pybind11;
    std::cout << "VoxelModel::createFromPcd() called with " << pcd.size() << " points.\n";

    TORCH_CHECK(global_scene_extent_ > 0.f, "global_scene_extent_ must be set (>0).");
    TORCH_CHECK(fixed_vox_size_   > 0.f, "fixed_vox_size_ must be set (>0).");
    TORCH_CHECK(max_sh_degree_ >= 0, "max_sh_degree_ must be >= 0.");

    const int N = static_cast<int>(pcd.size());
    if (N == 0) {
        std::cerr << "[createFromPcd] Empty PCD — nothing to do.\n";
        return;
    }

    // ------------------------------------------------------------------------
    // 0) Fixed global scene + desired voxel size
    // ------------------------------------------------------------------------
    auto dev = torch::kCUDA;
    scene_center_ = torch::tensor(
        {global_scene_center_[0], global_scene_center_[1], global_scene_center_[2]},
        torch::dtype(torch::kFloat32).device(dev)
    ).contiguous();                                                  // [3]
    // scene_extent_ = torch::tensor({global_scene_extent_},
    //     torch::dtype(torch::kFloat32).device(dev)
    // ).contiguous();   
    
    int   outside_level = 0;
    const float scene_extent_scalar = global_scene_extent_ * std::pow(2.0f, outside_level);
    scene_extent_ = torch::tensor({scene_extent_scalar},
        torch::dtype(torch::kFloat32).device(dev)).contiguous();
    inside_extent_ = torch::tensor({global_scene_extent_},
        torch::dtype(torch::kFloat32).device(dev)
    ).contiguous();    
        
    scene_min_t_  = (scene_center_ - 0.5f * scene_extent_).contiguous(); // [3]

    // Pack inputs
    torch::Tensor xyz = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    torch::Tensor rgb = torch::empty({N,3}, torch::dtype(torch::kFloat32).device(dev));
    {
        int i = 0;
        for (const auto& kv : pcd) {
            const auto& P = kv.second;
            xyz[i][0] = P.xyz_(0);
            xyz[i][1] = P.xyz_(1);
            xyz[i][2] = P.xyz_(2);
            // P.color_ already scaled? If it’s 0..255, divide; if already 0..1, keep.
            // std::cout << "Point " << i << " color before scaling: "
            //           << P.color_(0) << ", " << P.color_(1) << ", " << P.color_(2) << "\n";
            rgb[i][0] = P.color_(0);
            rgb[i][1] = P.color_(1);
            rgb[i][2] = P.color_(2);
            ++i;
        }
    }
    // ------------------------------------------------------------------------
    // 1) Compute octlevel from vox_size using utils.vox_size_2_level
    //    (mirror points_init behavior: round/clamp)
    // ------------------------------------------------------------------------
    py::gil_scoped_acquire gil;
    static py::module svr_utils = py::module::import("svraster_cuda").attr("utils");
    static py::module svm_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.sparse_voxel_model");
    }();
    static py::module act_mod   = py::module::import("src.utils.activation_utils");
    static py::module oct_utils = py::module::import("src.utils.octree_utils");

    const int MAX_L = max_num_levels_;
    // Level as float (tensor) then rounded like points_init (nearest by default)
    auto vox_size_t = torch::full({1}, fixed_vox_size_, torch::dtype(torch::kFloat32).device(dev)); // [1]
    py::object L_fp_py = oct_utils.attr("vox_size_2_level")(py::cast(scene_extent_), py::cast(vox_size_t));
    auto L_fp = L_fp_py.cast<torch::Tensor>().round();                                               // [1] float
    auto L_clamped = L_fp.clamp_min(1).clamp_max(MAX_L).to(torch::kInt8).contiguous();               // [1] int8

    // cache scalar level
    octlevel_ = L_clamped.item<int8_t>();
    // cache [1,1] effective voxel size
    py::object vox_eff_py = oct_utils.attr("level_2_vox_size")(
        py::cast(scene_extent_), 
        // level_2_vox_size expects [N,1]; give it [1,1] then keep it cached
        py::cast(L_clamped.view({1,1}))
    );
    vox_eff_ = vox_eff_py.cast<torch::Tensor>().view({1,1}).contiguous();    
    // std::cout << "[createFromPcd] Using octlevel=" << octlevel[0][0].item<int>()
    //           << " (vox_size=" << vox_eff[0][0].item<float>() << " m) for fixed_vox_size_="
    //           << fixed_vox_size_ << " m.\n";

    // ------------------------------------------------------------------------
    // 2) Compute ijk with this level/voxel size (mirror points_init)
    // ------------------------------------------------------------------------
    // ijk = ((xyz - scene_min) / vox_size).long()
    auto vox_effN  = vox_eff_.expand({N,1});                                                         // [N,1]
    torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(torch::kLong);                  // [N,3]

    auto octlevelN = torch::full({N,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    auto L_long    = octlevelN.to(torch::kLong);

    // In libtorch C++, pass a vector to cat
    std::vector<torch::Tensor> cat_inputs{ijk, L_long};
    auto ijkl = torch::cat(cat_inputs, /*dim=*/1);     

    // Use Python torch.unique(dim=0, return_inverse=True) (simple & robust)
    torch::Tensor ijkl_unq, invmap;
    {
        static py::module torch_mod = py::module::import("torch");
        py::tuple uniq = torch_mod.attr("unique")(
            py::cast(ijkl.contiguous()),
            py::arg("dim") = 0,
            py::arg("sorted") = true,
            py::arg("return_inverse") = true
        );
        ijkl_unq = uniq[0].cast<torch::Tensor>().contiguous(); // [Nu,4]
        invmap   = uniq[1].cast<torch::Tensor>().contiguous(); // [N]
    }

    torch::Tensor ijk_u, L_u; 
    auto parts = torch::split_with_sizes(ijkl_unq, {3, 1}, /*dim=*/1); 
    ijk_u = parts[0].contiguous(); L_u = parts[1].to(torch::kInt8).contiguous(); 
    L_u = L_u.to(torch::kInt8).contiguous(); // [Nu,1]
    int64_t Nu = ijk_u.size(0);

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    // rgb_u.index_add_(0, invmap, rgb);
    // auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    // rgb_u = rgb_u / counts;
    rgb_u.index_reduce_(
        /*dim=*/0,
        /*index=*/invmap,
        /*source=*/rgb,
        /*reduce=*/"mean",
        /*include_self=*/false
    );

    // Defensive bound check: ijk in [0, 2^L)
    const int L0 = L_u[0].item<int8_t>();               // all rows have same level
    const long limit = (1L << L0);
    TORCH_CHECK((ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>(),
                "Points below scene_min — enlarge global_scene_extent_ or filter.");
    TORCH_CHECK((ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
                (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>(),
                "Points exceed scene bounds — enlarge global_scene_extent_ or filter.");

    // ------------------------------------------------------------------------
    // 3) utils: ijk -> octpath (no constructor calls)
    // ------------------------------------------------------------------------
    py::object octpath_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_u.contiguous()),
                                                            py::cast(L_u.contiguous()));
    auto octpath = octpath_py.cast<torch::Tensor>().contiguous();            // [Nu,1] int64

    if (!cams.empty())
    {
        py::gil_scoped_acquire gil;
        static py::module oct_utils = py::module::import("src.utils.octree_utils");
        static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");

        // Decode voxel centers/sizes for filtering
        py::tuple dec = oct_utils.attr("octpath_decoding")(
            py::cast(octpath.contiguous()),
            py::cast(L_u.contiguous()),
            py::cast(scene_center_.contiguous()),
            py::cast(scene_extent_.contiguous())
        );
        at::Tensor vox_center = dec[0].cast<at::Tensor>(); // [Nu,3]
        at::Tensor vox_size   = dec[1].cast<at::Tensor>(); // [Nu,1]
        std::cout << "octpath dev="   << (octpath.is_cuda() ? "cuda" : "cpu") 
            << " dtype="        << octpath.dtype() << "\n";
        std::cout << "vox_center dev="<< (vox_center.is_cuda() ? "cuda" : "cpu")
                << " dtype="        << vox_center.dtype() << "\n";
        std::cout << "vox_size dev="  << (vox_size.is_cuda() ? "cuda" : "cpu")
                << " dtype="        << vox_size.dtype() << "\n";

        // Build Python list of MiniCams (ensure CUDA tensors)
        py::list py_cams;
        py::module torch_mod = py::module::import("torch");
        py::object py_cuda = torch_mod.attr("device")("cuda");
        auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
            if (py::hasattr(obj, name)) {
                py::object t = obj.attr(name);
                // Only tensors have .is_cuda/.to; defensive check:
                if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                    obj.attr(name) = t.attr("to")(py_cuda);  // or t.attr("to")("cuda")
                }
            }
        };
        for (const auto& c : cams) {
            py::object py_cam = MiniCam_to_py(c);
            // matrices
            move_attr_to_cuda_if_tensor(py_cam, "w2c");
            move_attr_to_cuda_if_tensor(py_cam, "c2w");
            // vectors used by mark_near / other helpers
            move_attr_to_cuda_if_tensor(py_cam, "position");
            move_attr_to_cuda_if_tensor(py_cam, "lookat");
            py_cams.append(py_cam);
        }

        auto Nu_before = octpath.size(0);
        // mark_max_samp_rate -> keep rate>0
        at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
            py_cams,
            py::cast(octpath),
            py::cast(vox_center),
            py::cast(vox_size)
        ).cast<at::Tensor>();

        at::Tensor kept = rate > 0;
        int64_t n_rate_pos = kept.sum().item<int64_t>();
        // optional near filtering
        const float near_thresh = 0.2f;
        int64_t n_near_hit = 0;
        if (near_thresh > 0.0f) {
            at::Tensor is_near = svr_mod.attr("mark_near")(
                py_cams,
                py::cast(octpath),
                py::cast(vox_center),
                py::cast(vox_size),
                py::float_(near_thresh)
            ).cast<at::Tensor>();
            kept = kept & (~is_near);
            n_near_hit = is_near.sum().item<int64_t>();
        }

        auto idx = torch::nonzero(kept).view({-1});
        int64_t K = idx.size(0);
        // Apply mask (all tensors together, no reshapes)
        if (K > 0 && K < octpath.size(0)) {
            octpath = octpath.index_select(0, idx).contiguous();   // [K,1]
            L_u     = L_u.index_select(0, idx).contiguous();       // [K,1] already
            rgb_u   = rgb_u.index_select(0, idx).contiguous();     // [K,3]
        }
        // Recompute sizes and assert consistency
        Nu = octpath.size(0);
        // Debug prints
        std::cout << "[filter] Nu_before=" << Nu_before
                << " rate>0=" << n_rate_pos
                << " near_hit=" << n_near_hit
                << " kept_final=" << Nu << std::endl;
    }

    // Create/reuse SVM (DO NOT call model_init/points_init/...)
    if (!py_->svm || py_->svm.is_none()) {
        py::object SVM = svm_mod.attr("SparseVoxelModel");
        py_->svm = SVM(py::arg("sh_degree") = max_sh_degree_);
    }

    // scene tensors (single fixed scene; no outside shells => inside_extent == scene_extent)
    py_->svm.attr("scene_center")  = scene_center_.contiguous();   // [3]
    py_->svm.attr("scene_extent")  = scene_extent_.contiguous(); // [1]
    // py_->svm.attr("inside_extent") = scene_extent_.contiguous(); // [1]
    py_->svm.attr("inside_extent") = inside_extent_.contiguous(); // [1]

    // Topology
    py_->svm.attr("octpath")  = octpath;                          // [Nu,1] int64
    py_->svm.attr("octlevel") = L_u.contiguous();
    // py_->svm.attr("octlevel") = L_u.view({Nu,1}).contiguous();    // [Nu,1] int8

    // ------------------------------------------------------------------------
    // 4) Initialize learnables directly
    // ------------------------------------------------------------------------
    // Subdivision priority
    auto subdiv_p = torch::ones({Nu,1}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();
    py_->svm.attr("_subdiv_p")     = subdiv_p.detach().requires_grad_(); 

    // SH-0 from fused RGB
    py::object sh0_dc_py = act_mod.attr("rgb2shzero")(py::cast(rgb_u.contiguous()));
    auto sh0_dc = sh0_dc_py.cast<torch::Tensor>().contiguous().requires_grad_(); // [Nu,3]
    py_->svm.attr("_sh0")          = sh0_dc.detach().requires_grad_(); 

    // Higher-degree SH zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs = torch::zeros({Nu, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev)).requires_grad_();
    py_->svm.attr("_shs")          = shs.detach().requires_grad_(); 

    // Active SH degree
    py_->svm.attr("active_sh_degree") = py::int_(std::min(max_sh_degree_, 3));

    // Grid link => allocate per-grid-point density
    torch::Tensor grid_pts_key = py_->svm.attr("grid_pts_key").cast<torch::Tensor>(); // [Mg,3] int64
    auto geo_grid = torch::full({grid_pts_key.size(0), 1}, -10.0f,
                                torch::dtype(torch::kFloat32).device(dev)).requires_grad_();
    py_->svm.attr("_geo_grid_pts") = geo_grid.detach().requires_grad_();

    // ------------------------------------------------------------------------
    // 5) Sync to C++ members
    // ------------------------------------------------------------------------
    auto fetch = [&](const char* name){
        return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
    };
    this->oct_path_      = fetch("octpath");
    this->oct_level_     = fetch("octlevel");
    this->center_        = fetch("vox_center");
    this->size_          = fetch("vox_size").squeeze(1);
    this->vox_size_inv_  = 1.0f / size_;
    this->grid_pts_key_  = fetch("grid_pts_key");
    this->vox_key_       = fetch("vox_key");

    // learnables
    this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    this->sh0_           = fetch("_sh0").requires_grad_(true);
    this->shs_           = fetch("_shs").requires_grad_(true);
    this->subdiv_p_      = fetch("_subdiv_p").requires_grad_(true);

    // stats buffer
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));

    // Register with your optimizer
    VOXEL_MODEL_TENSORS_TO_VEC

    // inc = 0 for the initial snapshot (or any monotonic counter you keep)
    const int64_t inc0 = 0;
    // 1) Log the global fixed scene AABB once
    // rrLogGlobalSceneAABB(inc0);
    // 2) Log ALL voxels after creation
    // rrLogVoxelBoxes(this->center_, this->size_, /*max_boxes_for_viz=*/10000000000, "vox_all", inc0);
}

void VoxelModel::increasePcd(std::vector<float> pcd_full,
                             std::vector<float> colors,
                             const int /*iteration*/, const std::vector<sv::MiniCam>& cams)
{
    namespace py = pybind11;
    static int64_t inc_counter = 1;  // or make it a member if you prefer
    const int Nf = static_cast<int>(pcd_full.size());
    if (Nf < 3 || colors.size() < 3) return;
    const int N = Nf / 3;
    std::cout << "VoxelModel::increasePcd() called with " << N << " points).\n";
    TORCH_CHECK(global_scene_extent_ > 0.f && fixed_vox_size_ > 0.f,
                "increasePcd: scene extent / fixed vox size not set.");
    TORCH_CHECK(py_->svm && !py_->svm.is_none(),
                "increasePcd: SVM not initialized; call createFromPcd first.");

    // ——— 0) Build CPU tensors from raw arrays, normalize RGB ————————
    torch::Tensor xyz_cpu = torch::from_blob(
        pcd_full.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone();

    torch::Tensor rgb_cpu = torch::from_blob(
        colors.data(), { (int64_t)N, 3 },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).clone().div_(255.0f);

    // Move to CUDA
    auto dev = torch::kCUDA;
    auto cpu = torch::kCPU;
    torch::Tensor xyz = xyz_cpu.to(dev);
    torch::Tensor rgb = rgb_cpu.to(dev);

    // Build the min/max AABB tensor once:
    at::Tensor aabb = torch::stack({
        torch::tensor({ global_scene_center_[0] - 0.5f*global_scene_extent_,
                        global_scene_center_[1] - 0.5f*global_scene_extent_,
                        global_scene_center_[2] - 0.5f*global_scene_extent_ }),
        torch::tensor({ global_scene_center_[0] + 0.5f*global_scene_extent_,
                        global_scene_center_[1] + 0.5f*global_scene_extent_,
                        global_scene_center_[2] + 0.5f*global_scene_extent_ })
    });
    // rrLogPointsAndAABB(/*iteration=*/0, xyz, rgb, aabb,
    //                 /*tag=*/"increase_raw", /*cap=*/10000000000,
    //                 /*log_points=*/true, /*log_box=*/false, /*inc=*/inc_counter);

    // --- 2) Compute octlevel from fixed_vox_size_ (same as createFromPcd)
    py::gil_scoped_acquire gil;
    static py::module svr_utils = py::module::import("svraster_cuda").attr("utils");
    static py::module svm_mod = []{
        py::module sys = py::module::import("sys");
        sys.attr("path").attr("insert")(0, "/home/dimitris/svraster");
        return py::module::import("src.sparse_voxel_model");
    }();
    static py::module act_mod   = py::module::import("src.utils.activation_utils");
    static py::module oct_utils = py::module::import("src.utils.octree_utils");
    static py::module torch_mod = py::module::import("torch");
    const int MAX_L = max_num_levels_;

    auto vox_effN  = vox_eff_.expand({N,1});                                                         // [N,1]
    // torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).floor().to(torch::kLong);                  // [N,3]
    torch::Tensor ijk = ((xyz - scene_min_t_) / vox_effN).to(torch::kLong); 

    auto octlevelN = torch::full({N,1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous(); // [N,1]
    auto L_long    = octlevelN.to(torch::kLong);

    // In libtorch C++, pass a vector to cat
    std::vector<torch::Tensor> cat_inputs{ijk, L_long};
    auto ijkl = torch::cat(cat_inputs, /*dim=*/1);                                                         // [N,4]

    torch::Tensor ijkl_unq, invmap;
    {
        py::tuple uniq = torch_mod.attr("unique")(
            py::cast(ijkl.contiguous()),
            py::arg("dim") = 0,
            py::arg("sorted") = true,
            py::arg("return_inverse") = true
        );
        ijkl_unq = uniq[0].cast<torch::Tensor>().contiguous(); // [Nu,4]
        invmap   = uniq[1].cast<torch::Tensor>().contiguous(); // [N]
    }
    torch::Tensor ijk_u, L_u;
    auto parts = torch::split_with_sizes(ijkl_unq, {3,1}, 1);
    ijk_u = parts[0].contiguous();                                                                    // [Nu,3]
    L_u   = parts[1].to(torch::kInt8).contiguous();                                                   // [Nu,1]
    int64_t Nu = ijk_u.size(0);

    // Defensive bound check: ijk in [0, 2^L)
    const int8_t L0 = L_u[0].item<int8_t>();      // all rows share the same level
    const long limit = (1L << L0);
    // TORCH_CHECK(
    //     (ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
    //     (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
    //     (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>(),
    //     "increasePcd: points below scene_min — enlarge global_scene_extent_ or filter.");
    // TORCH_CHECK(
    //     (ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
    //     (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
    //     (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>(),
    //     "increasePcd: points exceed scene bounds — enlarge global_scene_extent_ or filter.");
    const bool in_low =
    (ijk_u.index({torch::indexing::Slice(),0}) >= 0).all().item<bool>() &&
    (ijk_u.index({torch::indexing::Slice(),1}) >= 0).all().item<bool>() &&
    (ijk_u.index({torch::indexing::Slice(),2}) >= 0).all().item<bool>();
    const bool in_high =
        (ijk_u.index({torch::indexing::Slice(),0}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),1}) < limit).all().item<bool>() &&
        (ijk_u.index({torch::indexing::Slice(),2}) < limit).all().item<bool>();
    if (!(in_low && in_high)) {
        std::cout << "[increasePcd] OOB detected — reinitializing via createFromPcd().\n";
        // A) Compute scene from this batch using main_scene_bound_pcd_heuristic
        {
            py::gil_scoped_acquire gil;
            static py::module bound_utils = py::module::import("src.utils.bounding_utils");
            static py::module types       = py::module::import("types");
            static py::module torch_mod   = py::module::import("torch");

            // xyz_cpu is a CPU float32 tensor with shape [N,3]
            py::object ns = types.attr("SimpleNamespace")(
                "points"_a = py::cast(xyz_cpu.contiguous()).attr("numpy")()
            );
            py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(ns, py::float_(0.1f));
            // Extract center as a C++ vector (handles float32/float64 seamlessly)
            std::vector<double> c_vec = py::cast<std::vector<double>>(cr[0]);
            TORCH_CHECK(c_vec.size() == 3, "main_scene_bound_pcd_heuristic: center must have 3 elements");
            float radius = py::cast<float>(cr[1]);
            // Update global scene (cube side = 2*radius)
            global_scene_center_[0] = static_cast<float>(c_vec[0]);
            global_scene_center_[1] = static_cast<float>(c_vec[1]);
            global_scene_center_[2] = static_cast<float>(c_vec[2]);
            global_scene_extent_    = 2.0f * radius;
        }
        // B) Reset Python SVM so createFromPcd builds fresh state
        {
            py::gil_scoped_acquire gil;
            py_->svm = py::none();
        }
        // C) Convert current batch to a temporary map and call createFromPcd
        {
            std::map<point3D_id_t, Point3D> tmp;
            static point3D_id_t id_seed = 1;  // local seq; independent of COLMAP ids, etc.
            for (int i = 0; i < N; ++i) {
                Point3D P;
                // world coords
                P.xyz_(0) = pcd_full[3*i + 0];
                P.xyz_(1) = pcd_full[3*i + 1];
                P.xyz_(2) = pcd_full[3*i + 2];
                // colors[] are the original 0..255 values (floats)
                P.color_(0) = colors[3*i + 0];
                P.color_(1) = colors[3*i + 1];
                P.color_(2) = colors[3*i + 2];
                tmp.emplace(id_seed++, P);
            }
            createFromPcd(tmp, cams);
        }
        // Log and exit this call
        return;
    }

    auto rgb_u = torch::zeros({Nu,3}, torch::dtype(torch::kFloat32).device(dev));
    // rgb_u.index_add_(0, invmap, rgb);
    // auto counts = torch::bincount(invmap, {}, Nu).view({Nu,1}).clamp_min(1);
    // rgb_u = rgb_u / counts;
    rgb_u.index_reduce_(
        /*dim=*/0,
        /*index=*/invmap,
        /*source=*/rgb,
        /*reduce=*/"mean",
        /*include_self=*/false
    );

    // ── 4) Build octpath for this batch ─────────────────────────────────────
    py::object octpath_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_u), py::cast(L_u));
    auto octpath_new = octpath_py.cast<torch::Tensor>().contiguous();                                   // [Nu,1] int64

    // ---- (NEW) SVR-style filtering with cameras ----
    if (!cams.empty()) {
        py::gil_scoped_acquire gil;
        static py::module oct_utils = py::module::import("src.utils.octree_utils");
        static py::module svr_mod   = py::module::import("svraster_cuda").attr("renderer");
        static py::module torch_mod = py::module::import("torch");

        // Decode voxel centers/sizes
        py::tuple dec = oct_utils.attr("octpath_decoding")(
            py::cast(octpath_new),
            py::cast(L_u),
            py::cast(scene_center_.contiguous()),
            py::cast(scene_extent_.contiguous())
        );
        at::Tensor vox_center = dec[0].cast<at::Tensor>(); // [Nu,3] cuda
        at::Tensor vox_size   = dec[1].cast<at::Tensor>(); // [Nu,1] cuda

        // Build Python list of CUDA MiniCams
        py::list py_cams;
        py::object py_cuda = torch_mod.attr("device")("cuda");
        auto move_attr_to_cuda_if_tensor = [&](py::object& obj, const char* name){
            if (py::hasattr(obj, name)) {
                py::object t = obj.attr(name);
                if (py::hasattr(t, "is_cuda") && !py::bool_(t.attr("is_cuda"))) {
                    obj.attr(name) = t.attr("to")(py_cuda);
                }
            }
        };
        for (const auto& c : cams) {
            py::object py_cam = MiniCam_to_py(c);
            move_attr_to_cuda_if_tensor(py_cam, "w2c");
            move_attr_to_cuda_if_tensor(py_cam, "c2w");
            move_attr_to_cuda_if_tensor(py_cam, "position");
            move_attr_to_cuda_if_tensor(py_cam, "lookat");
            py_cams.append(py_cam);
        }

        auto Nu_before = octpath_new.size(0);

        // rate > 0
        at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
            py_cams, py::cast(octpath_new), py::cast(vox_center), py::cast(vox_size)
        ).cast<at::Tensor>();
        at::Tensor kept = rate > 0;
        int64_t n_rate_pos = kept.sum().item<int64_t>();

        // near filtering (same threshold you used in createFromPcd)
        const float near_thresh = 0.2f;
        int64_t n_near_hit = 0;
        if (near_thresh > 0.0f) {
            at::Tensor is_near = svr_mod.attr("mark_near")(
                py_cams, py::cast(octpath_new), py::cast(vox_center), py::cast(vox_size),
                py::float_(near_thresh)
            ).cast<at::Tensor>();
            kept = kept & (~is_near);
            n_near_hit = is_near.sum().item<int64_t>();
        }

        auto idx = torch::nonzero(kept).view({-1});
        int64_t K = idx.size(0);

        if (K == 0) {
            std::cout << "[increasePcd/filter] all candidates filtered out, nothing to add.\n";
            return;
        }

        if (K < octpath_new.size(0)) {
            // Apply mask to ALL aligned tensors
            octpath_new = octpath_new.index_select(0, idx).contiguous(); // [K,1]
            L_u         = L_u.index_select(0, idx).contiguous();         // [K,1]
            rgb_u       = rgb_u.index_select(0, idx).contiguous();       // [K,3]
        }

        Nu = octpath_new.size(0); // update Nu after filtering

        // Sanity
        TORCH_CHECK(L_u.sizes() == torch::IntArrayRef({Nu,1}), "L_u shape mismatch after filtering");
        TORCH_CHECK(rgb_u.sizes() == torch::IntArrayRef({Nu,3}), "rgb_u shape mismatch after filtering");

        std::cout << "[increasePcd/filter] Nu_before=" << Nu_before
                << " rate>0=" << n_rate_pos
                << " near_hit=" << n_near_hit
                << " kept_final=" << Nu << std::endl;
    }
    // ---- end filtering ----

    // ── 5) Dedup against existing voxels (across-batch) ─────────────────────
    auto octpath_old  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();                    // [No,1] int64
    auto octlevel_old = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();                   // [No,1] int8

    // Packed 1D key: (octpath<<8) | level
    auto key_new = octpath_new.view({-1}).to(torch::kInt64)
                    .mul(256)
                    .add(L_u.view({-1}).to(torch::kInt64));

    torch::Tensor new_mask;
    if (octpath_old.numel() == 0) {
        std::cout << "octpath_old is empty => all new.\n";
        new_mask = torch::ones({Nu}, torch::dtype(torch::kBool).device(dev));
    } else {
        auto key_old = octpath_old.view({-1}).to(torch::kInt64)
                        .mul(256)
                        .add(octlevel_old.view({-1}).to(torch::kInt64));
        py::object isin_py = torch_mod.attr("isin")(py::cast(key_new), py::cast(key_old));
        auto is_dup = isin_py.cast<torch::Tensor>().to(torch::kBool);                                    // [Nu]
        new_mask = ~is_dup;
    }
    // --- Prevent re-adding a parent at L=base_L when children (L > base_L) already exist ---
    {
        if (octpath_old.numel() > 0) {
            auto dev = octpath_old.device();
            const int MAX_L  = max_num_levels_;            // must match svraster MAX_NUM_LEVELS
            const int base_L = static_cast<int>(octlevel_);// the level used by createFromPcd()

            TORCH_CHECK(base_L >= 1 && base_L <= MAX_L,
                        "[increasePcd] base_L (octlevel_) out of range: ", base_L,
                        " with MAX_L=", MAX_L);

            // Existing voxels whose level is strictly finer than base_L
            auto Lold_i64    = octlevel_old.view({-1}).to(torch::kInt64);     // [No]
            auto has_children= (Lold_i64 > base_L);                           // [No] bool

            if (has_children.any().item<bool>()) {

                // Mask that clears all octant bits *below* base_L.
                // Bits per level = 3; for a node at level L, its octant sits at shift = 3*(MAX_L - L).
                // To keep bits down to base_L (inclusive), clear the lowest 3*(MAX_L - base_L) bits.
                const int levels_below  = std::max(0, MAX_L - base_L);
                const int bits_to_clear = 3 * levels_below;
                long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                long long keep_mask_ll  = ~lower_mask;

                auto keep_mask = torch::full(
                    {1},
                    static_cast<int64_t>(keep_mask_ll),
                    torch::TensorOptions().dtype(torch::kInt64).device(dev)
                );

                // Compute the ancestor-at-base_L octpaths for those finer voxels
                auto op_old_i64  = octpath_old.view({-1}).to(torch::kInt64);   // [No]
                auto op_anc_base = (op_old_i64 & keep_mask);                   // [No]

                // Keep only rows where L_old > base_L
                auto sel = torch::nonzero(has_children).view({-1});            // [K]
                op_anc_base = op_anc_base.index_select(0, sel);                // [K]

                // Build the ancestor keys at base_L: ((octpath_anc_base<<8) | base_L)
                auto key_children_as_parent = op_anc_base.mul(256)
                                            .add(torch::full_like(op_anc_base,
                                                                static_cast<int64_t>(base_L)));

                // Unique + sorted to make isin faster
                auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
                    TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
                    if (t.numel() <= 1) return t.contiguous();
                    auto sort_res = t.sort(/*dim=*/0);
                    auto sorted   = std::get<0>(sort_res);
                    using torch::indexing::Slice;
                    auto keep = torch::empty_like(sorted, torch::kBool);
                    keep.index_put_({0}, true);
                    auto neq = sorted.index({Slice(1, torch::indexing::None)})
                            != sorted.index({Slice(torch::indexing::None, -1)});
                    keep.index_put_({Slice(1, torch::indexing::None)}, neq);
                    auto idx = torch::nonzero(keep).view({-1});
                    return sorted.index_select(0, idx).contiguous();
                };
                key_children_as_parent = unique_sorted_1d(key_children_as_parent);

                // If a candidate NEW (octpath, base_L) matches any ancestor of an existing finer voxel,
                // then inserting that parent would collide with existing children later.
                py::object isin_py = torch_mod.attr("isin")(
                    py::cast(key_new), py::cast(key_children_as_parent));
                auto would_collide_parent = isin_py.cast<torch::Tensor>().to(torch::kBool);  // [Nu]

                // Remove those candidates from insertion
                new_mask = new_mask & (~would_collide_parent);
            }
        }
    }

    if (!new_mask.any().item<bool>()) {
        std::cout << "[increasePcd] No new voxels (all duplicates). Nothing appended.\n";
        return;
    }
    auto sel = torch::nonzero(new_mask).view({-1});                                                     // [Nk]
    auto octpath_add = octpath_new.index_select(0, sel);                                                // [Nk,1]
    auto L_add       = L_u.index_select(0, sel);                                                         // [Nk,1]
    auto rgb_add     = rgb_u.index_select(0, sel);                                                       // [Nk,3]
    const int Nk = sel.size(0);

    int64_t Nm_added = 0;  // count of artifact voxels added later
    // ── 6) Append topology (old preserved) ──────────────────────────────────
    py_->svm.attr("octpath")  = torch::cat({octpath_old,  octpath_add}, 0).contiguous();
    py_->svm.attr("octlevel") = torch::cat({octlevel_old, L_add},       0).contiguous();

    // ── 7) Append learnables for new rows ───────────────────────────────────
    // _subdiv_p
    auto subdiv_old = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
    auto subdiv_add = torch::ones({Nk,1}, torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old, subdiv_add}, 0)
                                .contiguous().detach().requires_grad_();

    // _sh0 from fused rgb
    py::object sh0_add_py = act_mod.attr("rgb2shzero")(py::cast(rgb_add.contiguous())); // [Nk,3]
    auto sh0_add = sh0_add_py.cast<torch::Tensor>().contiguous();
    auto sh0_old = py_->svm.attr("_sh0").cast<torch::Tensor>();
    py_->svm.attr("_sh0")      = torch::cat({sh0_old,    sh0_add   }, 0)
                                    .contiguous().detach().requires_grad_();

    // _shs zeros
    const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
    auto shs_old = py_->svm.attr("_shs").cast<torch::Tensor>();
    auto shs_add = torch::zeros({Nk, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev));
    py_->svm.attr("_shs")      = torch::cat({shs_old,    shs_add   }, 0)
                                .contiguous().detach().requires_grad_();

    // ── 8) Rebuild grid links; grow _geo_grid_pts only if grid expanded ─────
    // Trigger SVProperties recompute via property access
    torch::Tensor grid_pts_key_new = py_->svm.attr("grid_pts_key").cast<torch::Tensor>();               // [M,3] int64
    const int64_t M_prev = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
    const int64_t M_curr = grid_pts_key_new.size(0);

    // Prepare grow only when needed (group 0)
    torch::Tensor grow; // undefined by default
    if (M_curr > M_prev) {
        grow = torch::full({M_curr - M_prev, 1}, -10.0f,
                torch::dtype(torch::kFloat32).device(dev))
            .contiguous().detach().requires_grad_();
    }
    // ── 9) Append rows to optimizer param groups (and keep py svm in sync) ──
    // Important: only call for non-empty additions.
    if (grow.defined() && grow.size(0) > 0) {
        appendGroup_(/*group_idx=*/0, /*add_rows=*/grow, /*svm_field_name=*/"_geo_grid_pts",
                    &this->_geo_grid_pts_);
    }
    if (Nk > 0) {
        appendGroup_(/*group_idx=*/1, /*add_rows=*/sh0_add, /*svm_field_name=*/"_sh0",
                    &this->sh0_);
        appendGroup_(/*group_idx=*/2, /*add_rows=*/shs_add, /*svm_field_name=*/"_shs",
                    &this->shs_);
    }

    if (fill_empty_cells_) {
        // --- A) Build a bbox from the pcd via Python heuristic (median + density) ---
        // Use the raw batch on CPU (xyz_cpu); the function expects an object with a `.points` ndarray
        static py::module bound_utils = py::module::import("src.utils.bounding_utils");
        static py::module types       = py::module::import("types");
        static py::module torch_mod   = py::module::import("torch");

        py::object xyz_cpu_py = py::cast(xyz_cpu.contiguous());             // torch CPU tensor
        py::object ns         = types.attr("SimpleNamespace")(
                                "points"_a = xyz_cpu_py.attr("numpy")()); // pcd.points = numpy

        // Call Python heuristic
        // py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(
        //     ns, py::float_(0.1)   // expose a float member/const for this
        // );

        torch::Tensor center_t;   // CUDA tensor [3]
        float          radius_f;  // scalar radius
        try {
            // Try Python heuristic
            py::tuple cr = bound_utils.attr("main_scene_bound_pcd_heuristic")(ns, py::float_(0.1f));
            if (py::len(cr) == 2) {
                // cr[0]: numpy array (shape [3]), cr[1]: float
                py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]); // torch CPU
                center_t = center_t_py.cast<torch::Tensor>().to(torch::kCUDA).contiguous();
                radius_f = cr[1].cast<float>();
            } else {
                throw std::runtime_error("heuristic returned unexpected output");
            }
        } catch (...) {
            std::cout << "[increasePcd] fill_empty_cells_: Python heuristic failed, using C++ fallback.\n";
            // ----- Fallback: compute center/radius purely in C++ -----
            // center = mean of points (CPU -> CUDA)
            auto center_cpu = xyz_cpu.mean(/*dim=*/0, /*keepdim=*/false).contiguous();    // [3], CPU
            center_t = center_cpu.to(torch::kCUDA);
            // radius = max L2 distance to center, with a floor of 3 * vox_size
            torch::Tensor diffs = (xyz_cpu - center_cpu).contiguous();                   // [N,3], CPU
            torch::Tensor dists = diffs.norm(2, /*dim=*/1);                              // [N], CPU
            float maxdist = dists.numel() ? dists.max().item<float>() : 0.0f;
            radius_f = std::max(3.0f * vox_eff_.item<float>(), maxdist);
        }

        // Convert back to Torch (CUDA) so we can do all math in tensors
        // py::object center_t_py = torch_mod.attr("from_numpy")(cr[0]);
        // auto center_t = center_t_py.cast<torch::Tensor>().to(torch::kCUDA).contiguous();      // [3]
        // float radius_f = cr[1].cast<float>();
        auto radius_t  = torch::full({3}, radius_f, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        auto bb_min = (center_t - radius_t).contiguous();   // [3], world coords
        auto bb_max = (center_t + radius_t).contiguous();   // [3]

        // --- B) Convert bbox (world) -> dense ijk box at cached level/voxel size ---
        // Use cached scene_min_t_ and vox_eff_ (scalar size, but we kept it as [1,1]):
        auto vox_t = vox_eff_.mean()        // 0-dim CUDA float tensor
                        .view({1})
                        .repeat({3})       // [3]
                        .contiguous();     // CUDA float

        // Convert to indices, clamp to grid range [0, 2^L - 1]
        const int64_t grid_limit = (1LL << static_cast<int>(octlevel_)); // 2^L
        auto ijk_min = ((bb_min - scene_min_t_) / vox_t).floor().to(torch::kLong);
        auto ijk_max = ((bb_max - scene_min_t_) / vox_t).ceil().to(torch::kLong) - 1;
        // auto ijk_min = ((bb_min - scene_min_t_) / vox_t).to(torch::kLong);
        // auto ijk_max = ((bb_max - scene_min_t_) / vox_t).to(torch::kLong);
        // auto ijk_min = (((bb_min - scene_min_t_) / vox_t) - 0.5f).ceil().to(torch::kLong); // for not getting out of the bounding box
        // auto ijk_max = (((bb_max - scene_min_t_) / vox_t) - 0.5f).floor().to(torch::kLong);

        // Clamp to valid grid
        auto zero = torch::zeros_like(ijk_min);
        auto lim  = torch::full_like(ijk_max, grid_limit - 1);
        // auto lim  = torch::full_like(ijk_max, grid_limit);
        ijk_min = torch::maximum(ijk_min, zero);
        ijk_max = torch::minimum(ijk_max, lim);

        auto lens = (ijk_max - ijk_min + 1);        // [3] Long on CUDA
        int64_t nx = lens[0].item<int64_t>();
        int64_t ny = lens[1].item<int64_t>();
        int64_t nz = lens[2].item<int64_t>();
        long double Nc_est = (long double)nx * ny * nz;

        int64_t cap = std::max<int64_t>(1, max_artifact_cells_);    // set a sensible default (e.g., 200k)
        int64_t stride = 1;
        if (Nc_est > cap) {
            long double s = std::cbrt(Nc_est / (long double)cap);
            stride = std::max<int64_t>(1, (int64_t)std::ceil(s));
        }

        // If degenerate (all out), skip
        if ((ijk_min <= ijk_max).all().item<bool>()) {
            // --- C) Enumerate dense cells & set-diff vs current SVM topology ---
            auto ir = torch::arange(ijk_min[0].item<int64_t>(), ijk_max[0].item<int64_t>() + 1,
                                    stride, torch::dtype(torch::kLong).device(torch::kCUDA));
            auto jr = torch::arange(ijk_min[1].item<int64_t>(), ijk_max[1].item<int64_t>() + 1,
                                    stride, torch::dtype(torch::kLong).device(torch::kCUDA));
            auto kr = torch::arange(ijk_min[2].item<int64_t>(), ijk_max[2].item<int64_t>() + 1,
                                    stride, torch::dtype(torch::kLong).device(torch::kCUDA));
            auto grids = torch::meshgrid({ir, jr, kr}, /*indexing=*/"ij");
            auto ijk_box = torch::stack(
                { grids[0].contiguous().view({-1}),
                grids[1].contiguous().view({-1}),
                grids[2].contiguous().view({-1}) }, 1).contiguous(); // [Nc,3]

            if (max_artifact_cells_ > 0 && ijk_box.size(0) > max_artifact_cells_) {
                std::cout << "[increasePcd] fill_empty_cells_: limiting artifact cells"<< std::endl;
                ijk_box = ijk_box.index({torch::indexing::Slice(0, max_artifact_cells_)});
            }

            auto dev = ijk_box.device();
            auto L_box = torch::full({ijk_box.size(0),1}, octlevel_, torch::dtype(torch::kInt8).device(dev)).contiguous();

            // Build morton/octpath for those cells
            py::object octpath_box_py = svr_utils.attr("ijk_2_octpath")(py::cast(ijk_box), py::cast(L_box));
            auto octpath_box = octpath_box_py.cast<torch::Tensor>().contiguous(); // [Nc,1] int64
            
            // Compare with CURRENT topology (which already includes real additions above)
            auto octpath_cur  = py_->svm.attr("octpath").cast<torch::Tensor>().contiguous();
            auto octlevel_cur = py_->svm.attr("octlevel").cast<torch::Tensor>().contiguous();

            auto key_box = octpath_box.view({-1}).to(torch::kInt64).mul(256)
                        + L_box.view({-1}).to(torch::kInt64);
            auto key_cur = octpath_cur.view({-1}).to(torch::kInt64).mul(256)
                        + octlevel_cur.view({-1}).to(torch::kInt64);

            py::object isin_py = torch_mod.attr("isin")(py::cast(key_box), py::cast(key_cur));
            auto is_dup   = isin_py.cast<torch::Tensor>().to(torch::kBool);
            auto new_mask_box = ~is_dup;

            // --- Also drop base-level parents of already-subdivided regions (L_old > base_L) ---
            {
                const int MAX_L  = max_num_levels_;
                const int base_L = static_cast<int>(octlevel_);

                if (octpath_cur.numel() > 0) {
                    auto Lold_i64     = octlevel_cur.view({-1}).to(torch::kInt64);   // [No]
                    auto has_children = (Lold_i64 > base_L);                         // [No] bool

                    if (has_children.any().item<bool>()) {
                        // Build mask to zero out octant bits below base_L
                        const int levels_below  = std::max(0, MAX_L - base_L);
                        const int bits_to_clear = 3 * levels_below;
                        long long lower_mask    = (bits_to_clear > 0) ? ((1LL << bits_to_clear) - 1LL) : 0LL;
                        long long keep_mask_ll  = ~lower_mask;

                        auto dev = octpath_cur.device();
                        auto keep_mask = torch::full(
                            {1},
                            static_cast<int64_t>(keep_mask_ll),
                            torch::TensorOptions().dtype(torch::kInt64).device(dev)
                        );

                        // Compute ancestors at base_L for all L_old > base_L
                        auto op_old_i64  = octpath_cur.view({-1}).to(torch::kInt64);  // [No]
                        auto op_anc_base = (op_old_i64 & keep_mask);                  // [No]
                        auto sel_child   = torch::nonzero(has_children).view({-1});   // [K]
                        op_anc_base      = op_anc_base.index_select(0, sel_child);    // [K]

                        // Keys for those parents at base_L
                        auto key_children_as_parent = op_anc_base.mul(256)
                                                    .add(torch::full_like(op_anc_base,
                                                                        static_cast<int64_t>(base_L)));

                        // small helper
                        auto unique_sorted_1d = [](const at::Tensor& t)->at::Tensor {
                            TORCH_CHECK(t.dim() == 1, "unique_sorted_1d expects a 1-D tensor");
                            if (t.numel() <= 1) return t.contiguous();
                            auto sort_res = t.sort(0);
                            auto sorted   = std::get<0>(sort_res);
                            using torch::indexing::Slice;
                            auto keep = torch::empty_like(sorted, torch::kBool);
                            keep.index_put_({0}, true);
                            auto neq = sorted.index({Slice(1, torch::indexing::None)})
                                    != sorted.index({Slice(torch::indexing::None, -1)});
                            keep.index_put_({Slice(1, torch::indexing::None)}, neq);
                            auto idx = torch::nonzero(keep).view({-1});
                            return sorted.index_select(0, idx).contiguous();
                        };

                        key_children_as_parent = unique_sorted_1d(key_children_as_parent);

                        // Any base-level candidate that equals one of these parents must be skipped
                        py::object isin_py2 = torch_mod.attr("isin")(
                            py::cast(key_box), py::cast(key_children_as_parent));
                        auto would_collide_parent = isin_py2.cast<torch::Tensor>().to(torch::kBool); // [Nc]

                        new_mask_box = new_mask_box & (~would_collide_parent);
                    }
                }
            }

            // Proceed with the filtered candidates
            if (new_mask_box.any().item<bool>()) {
                auto sel = torch::nonzero(new_mask_box).view({-1});
                Nm_added = sel.size(0);

                bb_min_viz       = bb_min;         // [3] (CUDA)
                bb_max_viz       = bb_max;         // [3] (CUDA)
                sel_artifacts_viz= sel.clone();    // [Nm] indices into ijk_box
                ijk_box_viz      = ijk_box.clone();// [Nc,3] (CUDA)

                auto octpath_add2 = octpath_box.index_select(0, sel); // [Nm,1]
                auto L_add2       = L_box.index_select(0, sel);       // [Nm,1]

                // Append topology
                py_->svm.attr("octpath")  = torch::cat({octpath_cur,  octpath_add2}, 0).contiguous();
                py_->svm.attr("octlevel") = torch::cat({octlevel_cur, L_add2},       0).contiguous();

                // Prepare learnables for artifacts
                const int n_sh_rest = (max_sh_degree_ + 1)*(max_sh_degree_ + 1) - 1;
                auto shs_add2    = torch::zeros({Nm_added, n_sh_rest, 3}, torch::dtype(torch::kFloat32).device(dev)).contiguous();
                auto subdiv_add2 = torch::ones({Nm_added,1},             torch::dtype(torch::kFloat32).device(dev)).contiguous();

                auto rgb_add2 = torch::empty({Nm_added,3}, torch::dtype(torch::kFloat32).device(dev));
                rgb_add2.index_put_({torch::indexing::Slice(),0}, artifact_bg_rgb_[0]);
                rgb_add2.index_put_({torch::indexing::Slice(),1}, artifact_bg_rgb_[1]);
                rgb_add2.index_put_({torch::indexing::Slice(),2}, artifact_bg_rgb_[2]);

                py::object sh0_add2_py = act_mod.attr("rgb2shzero")(py::cast(rgb_add2.contiguous()));
                auto sh0_add2 = sh0_add2_py.cast<torch::Tensor>().contiguous();

                // Grid growth and optimizer-preserving appends
                auto grid_pts_key_new = py_->svm.attr("grid_pts_key").cast<torch::Tensor>();
                const int64_t M_prev = grid_pts_key_.defined() ? grid_pts_key_.size(0) : 0;
                const int64_t M_curr = grid_pts_key_new.size(0);
                if (M_curr > M_prev) {
                    auto grow = torch::full({M_curr - M_prev, 1}, -10.0f,
                                torch::dtype(torch::kFloat32).device(dev))
                                .contiguous().detach().requires_grad_();
                    appendGroup_(/*group_idx=*/0, grow, "_geo_grid_pts", &this->_geo_grid_pts_);
                }
                appendGroup_(/*group_idx=*/1, sh0_add2, "_sh0", &this->sh0_);
                appendGroup_(/*group_idx=*/2, shs_add2, "_shs", &this->shs_);

                // subdiv_p (not in optimizer groups)
                auto subdiv_old2 = py_->svm.attr("_subdiv_p").cast<torch::Tensor>();
                py_->svm.attr("_subdiv_p") = torch::cat({subdiv_old2, subdiv_add2}, 0)
                                            .contiguous().detach().requires_grad_();
                this->subdiv_p_ = py_->svm.attr("_subdiv_p").cast<torch::Tensor>().requires_grad_(true);

                // std::cout << "[increasePcd] Added " << Nm_added << " artifact voxels (pcd bbox).\n";
            }

            // Keep your existing “new voxels” logging in sync by adding Nm_added to Nk (as noted earlier)
            // ... (use last_added = Nk + Nm_added later)
        } // valid bbox
    } // fill_empty_cells_

    // ── 10) Pull back the rest for renderer (topology/derived fields) ───────
    auto fetch = [&](const char* name){
        return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
    };
    this->oct_path_      = fetch("octpath");
    this->oct_level_     = fetch("octlevel");
    this->center_        = fetch("vox_center");
    this->size_          = fetch("vox_size").squeeze(1);
    this->vox_size_inv_  = 1.0f / size_;
    this->grid_pts_key_  = fetch("grid_pts_key");
    this->vox_key_       = fetch("vox_key");
    // (optimizer params already set by appendGroup_; just ensure requires_grad)
    this->subdiv_p_      = py_->svm.attr("_subdiv_p").cast<torch::Tensor>().requires_grad_(true);
    // stats buffer resize
    this->max_w_ = torch::zeros({center_.size(0), 1}, torch::dtype(torch::kFloat32).device(dev));

    // ── 10) Re-register with optimizer (new rows appended) ───────────────────
    VOXEL_MODEL_TENSORS_TO_VEC

    // 1) Always (re)log the global scene box so timeline scrubbing shows context
    // rrLogGlobalSceneAABB(inc_counter);
    // 2) Log ALL voxels
    // rrLogVoxelBoxes(this->center_, this->size_, /*max_boxes_for_viz=*/10000000000, "vox_all", inc_counter);
    // 3) Log ONLY the newly-added voxels (orange)s
    const int64_t last_added = Nk + Nm_added;
    if (last_added > 0) {
        auto start = this->center_.size(0) - last_added;
        auto idx = torch::arange(start, this->center_.size(0),
                                this->center_.options().dtype(torch::kLong));
        auto centers_new = this->center_.index_select(0, idx);
        auto size_new    = this->size_.index_select(0, idx);
        // rrLogVoxelBoxes(centers_new, size_new, /*max*/10000000000, "vox_new", inc_counter);
    }

    // --- One-shot visualization of the algorithm ---
    static bool viz_once_done = false;
    if (fill_empty_cells_ && !viz_once_done && Nm_added > 0 &&
        bb_min_viz.defined() && bb_max_viz.defined() &&
        sel_artifacts_viz.defined() && ijk_box_viz.defined()) {

        std::vector<torch::Tensor> aabb_list{bb_min_viz, bb_max_viz};
        at::Tensor aabb = torch::stack(aabb_list).contiguous();

        rrLogPointsAndAABB(0, xyz, rgb, aabb,
            "viz/pcd_and_bbox", 1000000, /*log_points=*/true, /*log_box=*/true, inc_counter);

        auto ijk_art  = ijk_box_viz.index_select(0, sel_artifacts_viz).to(torch::kFloat32);
        auto vox_t_viz= vox_eff_.mean().view({1}).repeat({3});
        auto art_pts  = (scene_min_t_ + (ijk_art + 0.5f) * vox_t_viz).contiguous();

        auto art_rgb  = torch::empty({art_pts.size(0), 3},
                            torch::dtype(torch::kFloat32).device(art_pts.device()));
        art_rgb.index_put_({torch::indexing::Slice(),0}, artifact_bg_rgb_[0]);
        art_rgb.index_put_({torch::indexing::Slice(),1}, artifact_bg_rgb_[1]);
        art_rgb.index_put_({torch::indexing::Slice(),2}, artifact_bg_rgb_[2]);

        rrLogPointsAndAABB(0, art_pts, art_rgb, aabb,
            "viz/artifact_points", 200000, /*log_points=*/true, /*log_box=*/false, inc_counter);

        const int64_t total = this->center_.size(0);
        const int64_t last_added = Nk + Nm_added;
        if (last_added > 0 && total >= last_added) {
            auto optsL = this->center_.options().dtype(torch::kLong);
            auto idx_real_new = (Nm_added > 0)
                ? torch::arange(total - last_added, total - Nm_added, optsL)
                : torch::empty({0}, optsL);
            auto idx_art_new  = (Nm_added > 0)
                ? torch::arange(total - Nm_added, total, optsL)
                : torch::empty({0}, optsL);

            if (idx_real_new.numel() > 0) {
                auto c_real = this->center_.index_select(0, idx_real_new);
                auto s_real = this->size_.index_select(0,   idx_real_new);
                rrLogVoxelBoxes(c_real, s_real, 200000, "viz/vox_new_real", inc_counter);
            }
            if (idx_art_new.numel() > 0) {
                auto c_art = this->center_.index_select(0, idx_art_new);
                auto s_art = this->size_.index_select(0,   idx_art_new);
                rrLogVoxelBoxes(c_art, s_art, 200000, "viz/vox_new_artifacts", inc_counter);
            }
        }
        viz_once_done = true;
    }

    ++inc_counter;
    // std::cout << "[increasePcd] Appended " << Nk << " new voxels. Total now: "
    //         << this->oct_path_.size(0) << "\n";
    // std::cout << "[increasePcd] Appended " << (Nk + Nm_added)
    //       << " voxels this call (real=" << Nk
    //       << ", artifact=" << Nm_added << "). Total now: "
    //       << this->oct_path_.size(0) << "\n";
}

void VoxelModel::syncFromPython_() {
    py::gil_scoped_acquire gil;

    auto fetch = [&](const char* name){
        return py_->svm.attr(name).cast<torch::Tensor>().contiguous();
    };

    // Octree / indexing
    this->oct_path_   = fetch("octpath");
    this->oct_level_  = fetch("octlevel");
    this->vox_key_    = fetch("vox_key");               // [N,8] long
    this->center_     = fetch("vox_center");            // [N,3] float
    this->size_       = fetch("vox_size").squeeze(1);   // [N] or [N,1]→[N]
    this->vox_size_inv_  = 1.0f / this->size_;
    this->grid_pts_key_  = fetch("grid_pts_key");
    // Learnables
    this->_geo_grid_pts_ = fetch("_geo_grid_pts").requires_grad_(true);
    this->sh0_ = fetch("_sh0").requires_grad_(true);
    // this->sh0_ = this->sh0_.view({-1,1,3});
    this->shs_ = fetch("_shs").requires_grad_(true);
    this->subdiv_p_ = fetch("_subdiv_p").requires_grad_(true);
    // Resize any side buffers bound to voxel count
    this->max_w_ = torch::zeros({center_.size(0), 1},
              torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));

    VOXEL_MODEL_TENSORS_TO_VEC
}

void VoxelModel::appendGroup_(int group_idx,
                              const torch::Tensor& add_rows,
                              const char* svm_field_name,
                              torch::Tensor* out_member_param) {
    namespace py = pybind11;
    py::gil_scoped_acquire gil;

    if (!py_->optimizer_py || py_->optimizer_py.is_none()) return;

    // 1) param_groups[i]['params'][0]
    py::list groups = py_->optimizer_py.attr("param_groups");
    auto g = groups[group_idx].cast<py::dict>();
    py::list gparams = g["params"].cast<py::list>();
    torch::Tensor old_param = gparams[0].cast<torch::Tensor>();

    // 2) state dict (tensor-keyed)
    py::dict state = py_->optimizer_py.attr("state").cast<py::dict>();
    bool had_state = state.contains(py::cast(old_param));

    // 3) Build new concatenated param (cat along dim=0)
    // torch::Tensor new_param = torch::cat({old_param, add_rows}, /*dim=*/0).contiguous().requires_grad_(true);
    torch::Tensor new_param = torch::cat({old_param, add_rows}, /*dim=*/0).contiguous().detach().requires_grad_(true); // leaf

    // 4) Move/extend state if present
    if (had_state) {
        py::dict old_s = state[py::cast(old_param)].cast<py::dict>();
        // Copy step
        py::object step_obj = old_s["step"];
        // Extend exp_avg / exp_avg_sq
        torch::Tensor exp_avg    = old_s["exp_avg"].cast<torch::Tensor>();
        torch::Tensor exp_avg_sq = old_s["exp_avg_sq"].cast<torch::Tensor>();
        torch::Tensor z = torch::zeros_like(add_rows);

        torch::Tensor exp_avg_new    = torch::cat({exp_avg,    z}, 0).contiguous();
        torch::Tensor exp_avg_sq_new = torch::cat({exp_avg_sq, z}, 0).contiguous();

        // Construct new state
        py::dict new_s;
        new_s["step"]       = step_obj;         // keep same scalar
        new_s["exp_avg"]    = exp_avg_new;
        new_s["exp_avg_sq"] = exp_avg_sq_new;

        // Swap state key from old tensor to new tensor
        state.attr("__delitem__")(py::cast(old_param));
        state[py::cast(new_param)] = new_s;
    }
    // If no state existed yet, that's fine — SparseAdam lazily initializes on first .step()

    // 5) Rebind group param to the *same* new tensor object
    gparams[0] = py::cast(new_param);
    g["params"] = gparams;     // write back (usually not necessary, but explicit)

    // 6) Keep Python SVM and C++ members in sync with the *same* tensor object
    py_->svm.attr(svm_field_name) = new_param;
    *out_member_param = new_param;
}