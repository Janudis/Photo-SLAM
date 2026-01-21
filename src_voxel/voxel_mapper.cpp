#include "include_voxel/voxel_mapper.h"

namespace py = pybind11;
std::ofstream loss_log_;
std::ofstream loss_l1_log_;
std::ofstream loss_ssim_log_;
std::ofstream loss_l2_log_;

void saveTensor(const torch::Tensor &t,
                const std::string &tag,
                const std::string &dbg_dir,
                int iter,
                int image_id)
{
    auto img = t.squeeze(0)
                 .permute({1,2,0})
               // optional gamma:
               // .clamp(0.0f, 1.0f).pow(1.0f/2.2f)
                 .mul(255.0f).clamp(0.0f,255.0f)
                 .to(torch::kUInt8)
                 .contiguous()
                 .cpu();
    int H = img.size(0), W = img.size(1);
    cv::Mat rgb(H, W, CV_8UC3, img.data_ptr());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    std::ostringstream ss;
    ss << dbg_dir << "/" 
       << tag 
       << "_iter" << std::setw(6) << std::setfill('0') << iter
       << "_img"  << std::setw(3) << std::setfill('0') << image_id
       << ".png";
    cv::imwrite(ss.str(), bgr);
}

inline void saveDebugImage(torch::Tensor tensor, const std::string& path) {
    torch::NoGradGuard ng;
    namespace fs = std::filesystem;

    const fs::path out_path(path);
    if (!out_path.parent_path().empty())
        fs::create_directories(out_path.parent_path());

    // Normalize to (C,H,W)
    tensor = tensor.detach();
    if (tensor.dim() == 4 && tensor.size(0) == 1) tensor = tensor.squeeze(0); // (1,3,H,W)->(3,H,W)
    if (tensor.dim() == 2)                        tensor = tensor.unsqueeze(0); // (H,W)->(1,H,W)
    if (tensor.dim() == 3 && tensor.size(0) == 1) tensor = tensor.expand({3, tensor.size(1), tensor.size(2)});
    if (tensor.dim() != 3 || tensor.size(0) != 3) {
        std::cerr << "[saveDebugImage_fast] bad shape " << tensor.sizes() << "\n";
        return;
    }

    // GPU-side quantize to uint8 to reduce D2H bandwidth 4x
    if (tensor.device().is_cuda()) {
        if (tensor.dtype() != torch::kUInt8)
            tensor = tensor.clamp(0, 1).mul(255).to(torch::kUInt8); // still on GPU

        // Copy into pinned host memory
        auto cpu_u8 = torch::empty_like(
            tensor, tensor.options().device(torch::kCPU).pinned_memory(true));

        // For correctness keep this blocking; if you offload imwrite to a thread,
        // switch to non_blocking=true and synchronize appropriately there.
        cpu_u8.copy_(tensor, /*non_blocking=*/false);
        tensor = cpu_u8;
    } else {
        if (tensor.dtype() != torch::kUInt8)
            tensor = tensor.clamp(0, 1).mul(255).to(torch::kUInt8);
    }

    // HWC for OpenCV
    tensor = tensor.permute({1, 2, 0}).contiguous();
    cv::Mat rgb(tensor.size(0), tensor.size(1), CV_8UC3, tensor.data_ptr<uint8_t>());

    // Convert to BGR (skip this if your tensors are already BGR)
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

    // Encoder params
    std::vector<int> params;
    const std::string ext = out_path.extension().string();
    if (ext == ".png" || ext == ".PNG") {
        params = {cv::IMWRITE_PNG_COMPRESSION, 1};
    } else if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG") {
        params = {cv::IMWRITE_JPEG_QUALITY, 90};
    }

    if (!cv::imwrite(out_path.string(), bgr, params)) {
        std::cerr << "[saveDebugImage_fast] imwrite failed: " << out_path << "\n";
    }
}

static inline void extendAABB(Eigen::Vector3f& mn, Eigen::Vector3f& mx,
                              const Eigen::Vector3f& p) {
    mn = mn.cwiseMin(p);
    mx = mx.cwiseMax(p);
}

static inline void extendAABB_with_flat_xyz(Eigen::Vector3f& mn, Eigen::Vector3f& mx,
                                            const std::vector<float>& flat_xyz) {
    const size_t n = flat_xyz.size();
    if (n < 3) return;
    // If mn is not initialized yet, seed from the first triplet
    if (!std::isfinite(mn.x())) {
        mn = Eigen::Vector3f(flat_xyz[0], flat_xyz[1], flat_xyz[2]);
        mx = mn;
    }
    for (size_t i = 0; i + 2 < n; i += 3) {
        Eigen::Vector3f p(flat_xyz[i+0], flat_xyz[i+1], flat_xyz[i+2]);
        extendAABB(mn, mx, p);
    }
}

namespace {

std::mutex g_dumpkf_mutex;

static void saveKfPng_fromFloatRGB(const cv::Mat& im_float_rgb,   // CV_32FC3 in [0..1] RGB
                                   int fid,
                                   const std::filesystem::path& imgs_dir)
{
    // convert float[0..1] RGB -> 8-bit BGR
    cv::Mat tmp8, bgr8;
    im_float_rgb.convertTo(tmp8, CV_8UC3, 255.0);
    cv::cvtColor(tmp8, bgr8, cv::COLOR_RGB2BGR);
    std::ostringstream oss;
    oss << "kf_" << fid << ".png";
    const auto img_path = (imgs_dir / oss.str()).string();
    bool ok = cv::imwrite(img_path, bgr8);
    // std::cout << "[saveKfPng] " << img_path << " ok=" << std::boolalpha << ok
    //           << " mean(BGR)=" << cv::mean(bgr8) << std::endl;
}

// template<typename KFMap, typename CamContainer>
// void dumpKeyframesForProjectionFile(const KFMap& kfmap,
//                                     const CamContainer& cameras,
//                                     const std::filesystem::path& out_dir)
// {
//     std::lock_guard<std::mutex> lk(g_dumpkf_mutex);
//     std::filesystem::create_directories(out_dir);
//     std::filesystem::create_directories(out_dir / "imgs");
//     const auto tmp_file = out_dir / "keyframes_proj.tmp";
//     const auto out_file = out_dir / "keyframes_proj.txt";
//     std::ofstream os(tmp_file, std::ios::trunc);
//     if (!os) { std::cerr << "[dumpKF] cannot open " << tmp_file << '\n'; return; }
//     size_t n_lines = 0;
//     for (const auto& kv : kfmap) {
//         const auto& kf_ptr = kv.second;
//         if (!kf_ptr) continue;
//         const auto cam_id = kf_ptr->camera_id_;
//         auto cam_it = cameras.find(cam_id);
//         if (cam_it == cameras.end()) continue;
//         const sv::Camera& cam = cam_it->second;
//         const int   W  = kf_ptr->image_width_;
//         const int   H  = kf_ptr->image_height_;
//         const float fx = cam.fx(), fy = cam.fy(), cx = cam.cx(), cy = cam.cy();
//         const Eigen::Matrix4f Tcw = kf_ptr->getWorld2View2(kf_ptr->trans_, kf_ptr->scale_);
//         std::ostringstream oss;
//         oss << "imgs/kf_" << kf_ptr->fid_ << ".png";
//         const std::string rel_img = oss.str();
//         os << kf_ptr->fid_ << ' '
//            << W << ' ' << H << ' '
//            << std::setprecision(9) << fx << ' ' << fy << ' ' << cx << ' ' << cy << ' ';
//         for (int r = 0; r < 4; ++r)
//             for (int c = 0; c < 4; ++c)
//                 os << Tcw(r,c) << ' ';
//         os << rel_img << '\n';
//         ++n_lines;
//     }
//     os.close();
//     std::error_code ec;
//     std::filesystem::rename(tmp_file, out_file, ec);
//     if (ec) {
//         std::filesystem::copy_file(tmp_file, out_file,
//                                    std::filesystem::copy_options::overwrite_existing, ec);
//         std::filesystem::remove(tmp_file);
//     }
//     // std::cout << "[dumpKF] wrote " << out_file << " (" << n_lines << " lines)\n";
// }
} // namespace

inline void write_npy_float32(const std::string &path, const torch::Tensor &tensor)
{
    // Ensure tensor is contiguous and on CPU
    auto t = tensor.to(torch::kCPU).contiguous();

    if (t.dtype() != torch::kFloat32) {
        throw std::runtime_error("[write_npy_float32] Tensor must be float32.");
    }
    if (t.dim() < 1) {
        throw std::runtime_error("[write_npy_float32] Tensor must have at least 1 dimension.");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("[write_npy_float32] Failed to open file: " + path);
    }

    // Prepare NPY header
    // Magic string: \x93NUMPY
    const char magic[] = "\x93NUMPY";
    file.write(magic, 6);

    // Version 1.0
    unsigned char major = 1;
    unsigned char minor = 0;
    file.write(reinterpret_cast<char *>(&major), 1);
    file.write(reinterpret_cast<char *>(&minor), 1);

    // Build shape string, e.g. "(100, 3)"
    std::ostringstream shape_stream;
    shape_stream << "(";
    for (int i = 0; i < t.dim(); i++) {
        shape_stream << t.size(i);
        if (i < t.dim() - 1)
            shape_stream << ", ";
    }
    if (t.dim() == 1) {
        shape_stream << ",";
    }
    shape_stream << ")";

    // Little-endian, float32, fortran_order=False
    std::ostringstream header_stream;
    header_stream << "{'descr': '<f4', 'fortran_order': False, 'shape': "
                  << shape_stream.str() << ", }";
    std::string header = header_stream.str();

    // Pad header to 16-byte alignment
    int header_len = header.size() + 1; // +1 for newline
    int padding = 16 - ((10 + header_len) % 16);
    header.append(padding, ' ');
    header += "\n";

    // Write header length (2 bytes, little-endian)
    uint16_t header_size = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<char *>(&header_size), 2);

    // Write header
    file.write(header.c_str(), header.size());

    // Write raw data
    file.write(reinterpret_cast<const char *>(t.data_ptr()), t.numel() * sizeof(float));
    file.close();
}

inline torch::Tensor read_npy_float32(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("[read_npy_float32] Failed to open file: " + path);
    }

    // Check magic string
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("[read_npy_float32] Invalid NPY file magic string: " + path);
    }

    // Read version
    unsigned char major, minor;
    file.read(reinterpret_cast<char *>(&major), 1);
    file.read(reinterpret_cast<char *>(&minor), 1);
    if (!(major == 1 && minor == 0)) {
        throw std::runtime_error("[read_npy_float32] Only NPY v1.0 supported.");
    }

    // Read header length (little-endian uint16)
    uint16_t header_len;
    file.read(reinterpret_cast<char *>(&header_len), 2);

    // Read header content
    std::vector<char> header_buf(header_len);
    file.read(header_buf.data(), header_len);
    std::string header(header_buf.begin(), header_buf.end());

    // Parse shape from header
    auto pos1 = header.find("(");
    auto pos2 = header.find(")");
    if (pos1 == std::string::npos || pos2 == std::string::npos || pos2 <= pos1) {
        throw std::runtime_error("[read_npy_float32] Failed to parse shape.");
    }
    std::string shape_str = header.substr(pos1 + 1, pos2 - pos1 - 1);

    // Tokenize numbers in shape
    std::vector<int64_t> dims;
    std::stringstream ss(shape_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::stringstream num(item);
        int64_t val;
        if (num >> val)
            dims.push_back(val);
    }

    // Count elements
    int64_t numel = 1;
    for (auto d : dims)
        numel *= d;

    // Read data
    torch::Tensor tensor = torch::empty(dims, torch::kFloat32);
    file.read(reinterpret_cast<char *>(tensor.data_ptr()), numel * sizeof(float));
    file.close();

    return tensor;
}

static void save_initial_pcd_npy(
    const std::string& dir,
    const std::map<point3D_id_t, Point3D>& pcd)
{
    const int N = (int)pcd.size();
    torch::Tensor xyz = torch::empty({N,3}, torch::kFloat32);
    torch::Tensor rgb = torch::empty({N,3}, torch::kFloat32);
    torch::Tensor ids = torch::empty({N},   torch::kInt64);   // CPU

    int64_t* ids_ptr = ids.data_ptr<int64_t>();

    int i=0;
    for (const auto& kv : pcd) {
        const auto id = (int64_t)kv.first;
        const auto& P = kv.second;

        xyz.index_put_({i,0}, (float)P.xyz_(0));
        xyz.index_put_({i,1}, (float)P.xyz_(1));
        xyz.index_put_({i,2}, (float)P.xyz_(2));

        rgb.index_put_({i,0}, (float)P.color_(0));
        rgb.index_put_({i,1}, (float)P.color_(1));
        rgb.index_put_({i,2}, (float)P.color_(2));

        ids_ptr[i] = id;
        ++i;
    }

    // make sure the directory exists on your side

    write_npy_float32(dir + "/initial_xyz.npy", xyz);
    write_npy_float32(dir + "/initial_rgb.npy", rgb);
    torch::save(ids, dir + "/initial_ids.pt");
}

// pcd_log.h (continued)
static void log_increase_batch_npy(
    const std::string& dir,
    const std::vector<float>& points_flat,
    const std::vector<float>& colors_flat,
    int iter, int batch_idx)
{
    if (points_flat.size() % 3 != 0 || colors_flat.size() % 3 != 0)
        throw std::runtime_error("log_increase_batch_npy: flat vectors must be multiples of 3");

    const int Np = (int)(points_flat.size() / 3);
    const int Nc = (int)(colors_flat.size() / 3);
    if (Np != Nc)
        throw std::runtime_error("log_increase_batch_npy: points and colors count mismatch");

    // Ensure .../batches exists (C++17)
    #if __has_include(<filesystem>)
    #include <filesystem>
    std::filesystem::create_directories(dir + "/batches");
    #endif

    torch::Tensor xyz = torch::from_blob(
        const_cast<float*>(points_flat.data()),
        {Np, 3},
        torch::TensorOptions().dtype(torch::kFloat32))
        .clone(); // clone so we own memory

    torch::Tensor rgb = torch::from_blob(
        const_cast<float*>(colors_flat.data()),
        {Nc, 3},
        torch::TensorOptions().dtype(torch::kFloat32))
        .clone();

    char fn_xyz[512], fn_rgb[512], fn_meta[512];
    std::snprintf(fn_xyz,  sizeof(fn_xyz),  "%s/batches/xyz_%06d.npy",  dir.c_str(), batch_idx);
    std::snprintf(fn_rgb,  sizeof(fn_rgb),  "%s/batches/rgb_%06d.npy",  dir.c_str(), batch_idx);
    std::snprintf(fn_meta, sizeof(fn_meta), "%s/batches/meta_%06d.txt", dir.c_str(), batch_idx);

    write_npy_float32(fn_xyz, xyz);
    write_npy_float32(fn_rgb, rgb);

    std::ofstream m(fn_meta);
    m << "iter " << iter << "\n";
}

// Load initial + all batches/* (xyz_XXXXXX.npy + rgb_XXXXXX.npy)
static std::map<point3D_id_t, Point3D>
load_full_pcd_from_logs(const std::string& dir)
{
    std::map<point3D_id_t, Point3D> out;

    // --- 1) initial blobs ------------------------------------------------
    auto xyz0 = read_npy_float32(dir + "/initial_xyz.npy"); // [N,3] float32
    auto rgb0 = read_npy_float32(dir + "/initial_rgb.npy"); // [N,3] float32

    torch::Tensor ids0;
    torch::load(ids0, dir + "/initial_ids.pt");             // [N] int64
    auto ids_ptr = ids0.data_ptr<int64_t>();

    const int64_t N0 = xyz0.size(0);
    for (int64_t i = 0; i < N0; ++i) {
        Point3D P;
        // If Point3D uses Vector3d, keep (double) casts
        P.xyz_(0)   = (double)xyz0.index({i,0}).item<float>();
        P.xyz_(1)   = (double)xyz0.index({i,1}).item<float>();
        P.xyz_(2)   = (double)xyz0.index({i,2}).item<float>();
        P.color_(0) = (double)rgb0.index({i,0}).item<float>();
        P.color_(1) = (double)rgb0.index({i,1}).item<float>();
        P.color_(2) = (double)rgb0.index({i,2}).item<float>();

        point3D_id_t id = (point3D_id_t)ids_ptr[i];
        out[id] = P;
    }

    // Next ID for batch points (append-only)
    point3D_id_t next_id = out.empty() ? 0 : (std::prev(out.end())->first + 1);

    // --- 2) batches/* ----------------------------------------------------
    const std::string batches_dir = dir + "/batches";
    if (std::filesystem::exists(batches_dir)) {
        // Collect xyz files, sort by numeric index
        std::vector<std::pair<int, std::string>> xyz_files; // (idx, path)
        std::regex rex(R"(xyz_(\d+)\.npy)");

        for (auto& p : std::filesystem::directory_iterator(batches_dir)) {
            if (!p.is_regular_file()) continue;
            const auto name = p.path().filename().string();
            std::smatch m;
            if (std::regex_match(name, m, rex)) {
                int idx = std::stoi(m[1]);
                xyz_files.emplace_back(idx, p.path().string());
            }
        }
        std::sort(xyz_files.begin(), xyz_files.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });

        for (auto& [idx, xfile] : xyz_files) {
            // find matching rgb file
            std::string rfile = xfile;
            if (auto pos = rfile.rfind("xyz_"); pos != std::string::npos) {
                rfile.replace(pos, 3, "rgb");
            } else {
                // fallback: construct explicit path
                char buf[512];
                std::snprintf(buf, sizeof(buf), "%s/rgb_%06d.npy", batches_dir.c_str(), idx);
                rfile = buf;
            }

            // read
            auto xb = read_npy_float32(xfile);  // [Nb,3]
            auto rb = read_npy_float32(rfile);  // [Nb,3]
            const int64_t Nb = xb.size(0);

            // sanity: shapes must match
            if (rb.size(0) != Nb || xb.size(1) != 3 || rb.size(1) != 3) {
                throw std::runtime_error("Batch shape mismatch at index " + std::to_string(idx));
            }

            for (int64_t i = 0; i < Nb; ++i, ++next_id) {
                Point3D P;
                P.xyz_(0)   = (double)xb.index({i,0}).item<float>();
                P.xyz_(1)   = (double)xb.index({i,1}).item<float>();
                P.xyz_(2)   = (double)xb.index({i,2}).item<float>();
                P.color_(0) = (double)rb.index({i,0}).item<float>();
                P.color_(1) = (double)rb.index({i,1}).item<float>();
                P.color_(2) = (double)rb.index({i,2}).item<float>();
                out[next_id] = P;
            }
        }
    }

    return out;
}

// // --- header-scope constants (top of voxel_mapper.cpp) ---
// static constexpr int HMAP_R_MAX = 6;   // cap splat radius (pixels)
// static constexpr float Z_EPS = 1e-6f;
// static torch::Tensor approxGeomFromCentersAndSize(const sv::MiniCam& cam,
//                                                   const torch::Tensor& vox_center, // [N,3], CPU ok
//                                                   const torch::Tensor& vox_size,   // [N],   CPU ok
//                                                   int H, int W)
// {
//     auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
//     auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
//     torch::Tensor geom = torch::full({H, W}, (int64_t)-1, opts_i64);
//     torch::Tensor zbuf = torch::full({H, W}, std::numeric_limits<float>::infinity(), opts_f32);

//     torch::Tensor centers = vox_center.detach().to(torch::kCPU).contiguous();
//     torch::Tensor sizes   = vox_size.detach().to(torch::kCPU).contiguous();

//     torch::Tensor w2c = cam.w2c.detach().to(torch::kCPU).contiguous(); // [4,4]
//     const float fx = cam.fx, fy = cam.fy, cx = cam.cx, cy = cam.cy;

//     int64_t N = centers.size(0);
//     auto xyz1 = torch::cat({centers, torch::ones({N,1}, opts_f32)}, 1);
//     auto cam_xyz1 = torch::matmul(xyz1, w2c.t());
//     auto X = cam_xyz1.index({torch::indexing::Slice(), 0});
//     auto Y = cam_xyz1.index({torch::indexing::Slice(), 1});
//     auto Z = cam_xyz1.index({torch::indexing::Slice(), 2});

//     auto X_a = X.data_ptr<float>();
//     auto Y_a = Y.data_ptr<float>();
//     auto Z_a = Z.data_ptr<float>();
//     auto S_a = sizes.data_ptr<float>();
//     auto zbuf_a = zbuf.data_ptr<float>();
//     auto geom_a = geom.data_ptr<int64_t>();

//     auto in_bounds = [&](int u, int v) { return (u>=0 && u<W && v>=0 && v<H); };

//     for (int64_t i = 0; i < N; ++i) {
//         float z = Z_a[i];
//         if (z <= cam.near) continue;
//         float u_f = fx * (X_a[i] / z) + cx;
//         float v_f = fy * (Y_a[i] / z) + cy;

//         // projected *radius* in pixels ~ 0.5 * f * (size/z)
//         int ru = std::max(1, (int)std::ceil(0.5f * fx * (S_a[i] / std::abs(z))));
//         int rv = std::max(1, (int)std::ceil(0.5f * fy * (S_a[i] / std::abs(z))));

//         int u0 = (int)std::floor(u_f + 0.5f);
//         int v0 = (int)std::floor(v_f + 0.5f);

//         for (int dv = -rv; dv <= rv; ++dv) {
//             int vv = v0 + dv; if (vv < 0 || vv >= H) continue;
//             for (int du = -ru; du <= ru; ++du) {
//                 int uu = u0 + du; if (uu < 0 || uu >= W) continue;
//                 // optional: circular mask
//                 if ((du*du)/(float)(ru*ru) + (dv*dv)/(float)(rv*rv) > 1.0f) continue;
//                 int idx = vv * W + uu;
//                 if (z < zbuf_a[idx]) { zbuf_a[idx] = z; geom_a[idx] = i; }
//             }
//         }
//     }
//     return geom; // int64 [H,W], -1 empty
// }

VoxelMapper::VoxelMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                         const std::filesystem::path& config_file_path,
                        std::filesystem::path result_dir,
                        int seed,
                        torch::DeviceType device_type)
    : mpSLAM(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    std::srand(seed);
    torch::manual_seed(seed);
    loss_log_.open("loss.csv", std::ios::out);
    loss_ssim_log_.open("loss_ssim.csv", std::ios::out);
    loss_l1_log_.open("loss_l1.csv", std::ios::out);
    loss_l2_log_.open("loss_l2.csv", std::ios::out);

    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[VoxelMapper] CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        mDevice = torch::Device(torch::kCUDA);
        model_params_.data_device_ = "cuda";
    } else {
        std::cout << "[VoxelMapper] Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        mDevice = torch::Device(torch::kCPU);
        model_params_.data_device_ = "cpu";
    }

    // result_dir_ = mOutDir;
    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir);
    config_file_path_ = config_file_path;
    readConfigFromFile(config_file_path);

    // Background & override color setup
    std::vector<float> bg_color = {0.0f, 0.0f, 0.0f};  // no white-background logic needed
    if (model_params_.white_background_)
         bg_color = {1.0f, 1.0f, 1.0f};
     else
         bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    voxel_model_ = std::make_shared<sv::VoxelModel>(model_params_);
    scene_       = std::make_shared<sv::VoxelScene>(model_params_);
    
    size_t N = scene_->keyframes().size();
    best_loss_per_kf_.assign(N,  std::numeric_limits<float>::infinity());
    worst_loss_per_kf_.assign(N, -std::numeric_limits<float>::infinity());
    extrema_dir_ = result_dir_ / "extrema";
    {
        std::error_code ec;
        std::filesystem::remove_all(extrema_dir_, ec); // best-effort cleanup
    }
    std::filesystem::create_directories(extrema_dir_);

    switch (pSLAM->getSensorType()) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBD;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // /* Load every ORB-SLAM3 camera, convert to Camera, pre–compute            */
    auto settings = pSLAM->getSettings();   
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParams undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );
    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto& SLAM_camera : vpCameras) {
        sv::Camera camera;
        camera.camera_id_ = SLAM_camera->GetId();
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(sv::Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            // Undistortion
            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                    cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                    cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREO) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(sv::Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(sv::Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }
}

void VoxelMapper::readConfigFromFile(const std::filesystem::path& cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string(), cv::FileStorage::READ);
    if (!settings_file.isOpened()) {
        std::cerr << "[VoxelMapper] Failed to open cfg: " << cfg_path << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[VoxelMapper] Reading parameters from " << cfg_path << '\n';
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
     model_params_.sh_degree_ =
         settings_file["Model.sh_degree"].operator int();
     model_params_.resolution_ =
         settings_file["Model.resolution"].operator float();
     model_params_.white_background_ =
         (settings_file["Model.white_background"].operator int()) != 0;
     model_params_.eval_ =
         (settings_file["Model.eval"].operator int()) != 0;

    /* ───────── PIPELINE FLAGS ───────── */
    z_near_ =
         settings_file["Camera.z_near"].operator float();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    min_num_initial_map_kfs_ =
        static_cast<std::size_t>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ =
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    local_BA_increased_times_of_use_ = 
         settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
         settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();

    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();

    pipe_params_.convert_SHs_ =
         (settings_file["Pipeline.convert_SHs"].operator int()) != 0;

    do_gaus_pyramid_training_ =
         (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }
    
    /* ───────── OPTIMIZATION PARAMETERS ───────── */
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.geo_lr_ =
        settings_file["Optimization.geo_lr"].operator float();
    opt_params_.sh0_lr_ =
        settings_file["Optimization.sh0_lr"].operator float();
    opt_params_.shs_lr_ =
        settings_file["Optimization.shs_lr"].operator float();
    {
        cv::FileNode n = settings_file["Optimization.lr_decay_ckpt"];
        opt_params_.lr_decay_ckpt_.clear();
        if (!n.empty())
        {
            if (n.type() == cv::FileNode::SEQ) {
                // YAML: Optimization.lr_decay_ckpt: [5000, 10000, 20000]
                for (auto it = n.begin(); it != n.end(); ++it)
                    opt_params_.lr_decay_ckpt_.push_back((int)*it);
            } else if (n.isInt()) {
                // YAML: Optimization.lr_decay_ckpt: 10000
                opt_params_.lr_decay_ckpt_.push_back((int)n);
            } else if (n.isString()) {
                // YAML: Optimization.lr_decay_ckpt: "5000,10000,20000"
                std::string s = (std::string)n;
                std::stringstream ss(s);
                for (std::string tok; std::getline(ss, tok, ','); ) {
                    if (!tok.empty()) opt_params_.lr_decay_ckpt_.push_back(std::stoi(tok));
                }
            }
        }
    }
    opt_params_.optim_beta1_ =
        settings_file["Optimization.optim_beta1"].operator float();
    opt_params_.optim_beta2_ =
        settings_file["Optimization.optim_beta2"].operator float();
    opt_params_.optim_eps_ =
        settings_file["Optimization.optim_eps"].operator float();
    opt_params_.lr_decay_mult_ =
        settings_file["Optimization.lr_decay_mult"].operator float();

    opt_params_.adapt_from_ =
        settings_file["Optimization.adapt_from"].operator int();
    opt_params_.adapt_every_ =
        settings_file["Optimization.adapt_every"].operator int();
    opt_params_.prune_until_ =
        settings_file["Optimization.prune_until"].operator int();
    opt_params_.prune_thres_init_ =
        settings_file["Optimization.prune_thres_init"].operator float();
    opt_params_.prune_thres_final_ =
        settings_file["Optimization.prune_thres_final"].operator float();

    opt_params_.subdivide_until_ =
        settings_file["Optimization.subdivide_until"].operator int();
    opt_params_.subdivide_all_until_ =
        settings_file["Optimization.subdivide_all_until"].operator int();
    opt_params_.subdivide_samp_thres_ =
        settings_file["Optimization.subdivide_samp_thres"].operator float();
    opt_params_.subdivide_prop_ =
        settings_file["Optimization.subdivide_prop"].operator float();
    opt_params_.subdivide_max_num_ =
        settings_file["Optimization.subdivide_max_num"].operator int();

    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();

    opt_params_.lambda_tv_density_ =
        settings_file["Optimization.lambda_tv_density"].operator float();
    opt_params_.tv_from_ =
        settings_file["Optimization.tv_from"].operator int();
    opt_params_.tv_until_ =
        settings_file["Optimization.tv_until"].operator int();

    opt_params_.ss_aug_max_ = settings_file["Optimization.ss_aug_max"].operator float();
    opt_params_.lambda_R_concen_ = settings_file["Optimization.lambda_R_concen"].operator float();
    opt_params_.lambda_dist_ = settings_file["Optimization.lambda_dist"].operator float();
    opt_params_.lambda_T_inside_ = settings_file["Optimization.lambda_T_inside"].operator float();
    opt_params_.lambda_ssim_ = settings_file["Optimization.lambda_ssim"].operator float();

    opt_params_.lambda_sparse_depth_ = settings_file["Optimization.lambda_sparse_depth"].operator float();
    opt_params_.sparse_depth_until_ = settings_file["Optimization.sparse_depth_until"].operator int();

    /* ───────── LOGGING PARAMETERS ───────── */
    training_report_interval_ =
        settings_file["Record.training_report_interval"].operator int();
    keyframe_record_interval_ =
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ =
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ =
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ =
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ =
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    // Viewer Parameters
     rendered_image_viewer_scale_ =
         settings_file["VoxelViewer.image_scale"].operator float();
     rendered_image_viewer_scale_main_ =
         settings_file["VoxelViewer.image_scale_main"].operator float();

    // std::cout << "\n[CFG] Parsed Optimization Parameters:" << std::endl;
    // // std::cout << "  lr:                       " << opt_params_.position_lr_final_ << std::endl;
}

// sv::Camera  →  nvblox::Camera
inline nvblox::Camera toNvbloxCamera(const VoxelKeyframe& kf)
{
    const sv::Camera& cam = kf.cam_;
    // These should be the full undistorted image resolution.
    const int width  = kf.image_width_;
    const int height = kf.image_height_;

    return nvblox::Camera(
        cam.fx(), cam.fy(),   // fu, fv
        cam.cx(), cam.cy(),   // cu, cv
        width, height         // width, height
    );
}

// VoxelKeyframe (Tcw_) → nvblox::Transform T_L_C  (camera → layer/world)
inline nvblox::Transform toNvbloxTransform(VoxelKeyframe& kf)
{
    // Tcw_ is world→cam  ⇒  Twc = Tcw_.inverse() is cam→world
    Sophus::SE3f Tcw_f = kf.getPosef();       // SE3f world→cam
    Sophus::SE3f Twc_f = Tcw_f.inverse();     // SE3f cam→world

    nvblox::Transform T_L_C;
    T_L_C.linear()      = Twc_f.rotationMatrix();
    T_L_C.translation() = Twc_f.translation();
    return T_L_C;  // layer frame L == world frame W
}

// cv::Mat (CV_32FC1, meters) → nvblox::DepthImage (device)
inline void cvDepthToNvbloxDepth(const cv::Mat& depth_meters,
                                 nvblox::DepthImage* depth_img)
{
    using namespace nvblox;
    CHECK(depth_meters.type() == CV_32FC1);

    const int rows = depth_meters.rows;
    const int cols = depth_meters.cols;

    // Allocate / resize on device
    *depth_img = DepthImage(rows, cols, MemoryType::kDevice);

    DepthImage host_img(rows, cols, MemoryType::kHost);
    for (int r = 0; r < rows; ++r) {
        const float* src_row = depth_meters.ptr<float>(r);
        for (int c = 0; c < cols; ++c) {
            host_img(r, c) = src_row[c];
        }
    }

    // Host → device (synchronous)
    depth_img->copyFrom(host_img);
}

// Build an NVBlox camera whose resolution matches the depth image,
// and whose intrinsics are a scaled version of sv::Camera intrinsics.
inline nvblox::Camera makeNvbloxCameraFromDepthAndSvCam(
    const cv::Mat& depth_meters,
    const sv::Camera& cam)
{
    const int width  = depth_meters.cols;
    const int height = depth_meters.rows;

    CHECK(width  > 0);
    CHECK(height > 0);

    // Start from the Photo-SLAM camera intrinsics.
    float fx = cam.fx();
    float fy = cam.fy();
    float cx = cam.cx();
    float cy = cam.cy();

    const float cam_w = static_cast<float>(cam.width());
    const float cam_h = static_cast<float>(cam.height());

    // If Photo-SLAM uses a different internal resolution (e.g. 318x255),
    // rescale intrinsics to match the actual depth image size.
    if (cam_w > 0.0f && cam_h > 0.0f &&
        (static_cast<int>(cam_w) != width ||
         static_cast<int>(cam_h) != height))
    {
        const float scale_x = static_cast<float>(width)  / cam_w;
        const float scale_y = static_cast<float>(height) / cam_h;

        fx *= scale_x;
        fy *= scale_y;
        cx *= scale_x;
        cy *= scale_y;

        // std::cout << "[NVBLOX] Rescaling intrinsics for depth image: "
        //           << "cam(" << cam.width() << "x" << cam.height()
        //           << ") -> img(" << width << "x" << height << ") "
        //           << "scale_x=" << scale_x << " scale_y=" << scale_y
        //           << std::endl;
    }

    return nvblox::Camera(fx, fy, cx, cy, width, height);
}

void VoxelMapper::initializeNvbloxMapper()
{
    using namespace nvblox;

    // 1) Decide voxel size (from config ideally)
    sdf_voxel_size_m_ = 0.05f;  // keep or take from YAML

    // 2) Configure where TSDF lives (device is standard)
    BlockMemoryPoolParams pool_params;
    pool_params.memory_type = MemoryType::kDevice;

    // 3) Create mapper that integrates TSDF (no freespace / occupancy)
    auto cuda_stream = std::make_shared<CudaStreamOwning>();
    sdf_mapper_ = std::make_shared<Mapper>(
        sdf_voxel_size_m_,
        pool_params,
        ProjectiveLayerType::kTsdf,   // TSDF only
        cuda_stream
    );

    // 4) Set mapper params (defaults + small tweaks)
    MapperParams mapper_params;  // default-constructed

    mapper_params.esdf_integrator_params.esdf_integrator_max_distance_m = 5.0f;
    mapper_params.projective_integrator_params
        .projective_integrator_max_integration_distance_m = 4.0f;

    sdf_mapper_->setMapperParams(mapper_params);

    // Now override appearance integrator settings explicitly.
    auto& color_int = sdf_mapper_->color_integrator();
    color_int.sphere_tracing_ray_subsampling_factor(1);
    color_int.view_calculator().raycast_subsampling_factor(1);

    // (Optional) if you ever use feature integration:
    // auto& feat_int = sdf_mapper_->feature_integrator();
    // feat_int.sphere_tracing_ray_subsampling_factor(1);
    // feat_int.view_calculator().raycast_subsampling_factor(1);
    // std::cout << "[NVBLOX] color_integrator sphere_tracing_ray_subsampling_factor = "
    //           << color_int.sphere_tracing_ray_subsampling_factor() << "\n";
    // std::cout << "[NVBLOX] color_integrator view_calculator.raycast_subsampling_factor = "
    //           << color_int.view_calculator().raycast_subsampling_factor() << "\n";

    // --- DEBUG: print TSDF decay free distance and approximate truncation ---
    {
        nvblox::TsdfDecayIntegrator tsdf_decay;
        nvblox::FreespaceIntegrator freespace;
        std::cout << "[TEST] TsdfDecayIntegrator.free_distance_vox() = "
                << tsdf_decay.free_distance_vox() << " vox\n";
        std::cout << "[TEST] FreespaceIntegrator.max_tsdf_distance_for_occupancy_m() = "
                << freespace.max_tsdf_distance_for_occupancy_m() << " m\n";
    // // After sdf_mapper_ is constructed and setMapperParams() has been called:
    //     auto& decay = sdf_mapper_->tsdf_decay_integrator();
    //     std::cout << "[NVBLOX] TsdfDecayIntegrator.free_distance_vox() = "
    //               << decay.free_distance_vox() << " vox\n";

    //     auto& freespace = sdf_mapper_->freespace_integrator();
    //     std::cout << "[NVBLOX] FreespaceIntegrator.max_tsdf_distance_for_occupancy_m() = "
    //               << freespace.max_tsdf_distance_for_occupancy_m() << " m\n";
    }
}

void cvRgbToNvbloxColor(const cv::Mat& rgb_image,
                        nvblox::ColorImage* color_img,
                        nvblox::CudaStream* /*stream*/)
{
    CHECK(color_img != nullptr);
    CHECK(!rgb_image.empty());

    cv::Mat rgb_u8;
    if (rgb_image.type() == CV_8UC3) {
        rgb_u8 = rgb_image;
    } else {
        // Photo-SLAM often stores float images; convert to [0,255]
        rgb_image.convertTo(rgb_u8, CV_8UC3, 255.0);
    }

    const int rows = rgb_u8.rows;
    const int cols = rgb_u8.cols;

    // Host buffer
    nvblox::ColorImage color_host(rows, cols, nvblox::MemoryType::kHost);
    for (int v = 0; v < rows; ++v) {
        const cv::Vec3b* row_ptr = rgb_u8.ptr<cv::Vec3b>(v);
        for (int u = 0; u < cols; ++u) {
            const cv::Vec3b& c = row_ptr[u];
            nvblox::Color col;

            // OpenCV BGR → NVBlox RGB
            col.r() = c[2];
            col.g() = c[1];
            col.b() = c[0];

            color_host(v, u) = col;
        }
    }

    // Allocate device image and copy
    *color_img = nvblox::ColorImage(rows, cols, nvblox::MemoryType::kDevice);
    // copyFrom(ImageBase) – no stream parameter in this overload
    color_img->copyFrom(color_host);
}

void VoxelMapper::integrateKeyframeIntoNvblox(
    VoxelKeyframe& kf,
    const cv::Mat& depth_meters)
{
    if (!sdf_mapper_) {
        return;
    }

    const sv::Camera& cam = kf.cam_;

    // Depth resolution drives NVBlox camera resolution
    const int depth_w = depth_meters.cols;
    const int depth_h = depth_meters.rows;

    if (depth_w <= 0 || depth_h <= 0) {
        std::cout << "[NVBLOX] Warning: depth_meters has invalid size ("
                  << depth_w << "x" << depth_h << "), skipping integration.\n";
        return;
    }

    // Just for debugging, look at both RGB and depth sizes
    const int img_w = kf.img_undist_.cols;
    const int img_h = kf.img_undist_.rows;

    // std::cout << "[NVBLOX] integrate KF: "
    //           << "depth " << depth_w << "x" << depth_h
    //           << "  rgb "   << img_w   << "x" << img_h
    //           << "  cam.width()="  << cam.width()
    //           << " cam.height()=" << cam.height()
    //           << std::endl;

    // Build NVBlox camera whose resolution matches the depth image
    nvblox::Camera nvb_cam = makeNvbloxCameraFromDepthAndSvCam(depth_meters, cam);

    // 3) Pose (camera → world == layer)
    nvblox::Transform T_L_C = toNvbloxTransform(kf);

    // 4) Depth image → NVBlox
    static nvblox::DepthImage depth_img(nvblox::MemoryType::kDevice);
    cvDepthToNvbloxDepth(depth_meters, &depth_img);

    // Integrate TSDF
    sdf_mapper_->integrateDepth(depth_img, T_L_C, nvb_cam);

    // --- Color integration (optional) ---
    const cv::Mat& rgb_undistorted = kf.img_undist_;  // parallel color image

    if (!rgb_undistorted.empty()) {
        // It is nice (but not strictly required) that RGB matches depth size
        if (rgb_undistorted.cols != depth_w || rgb_undistorted.rows != depth_h) {
            std::cout << "[NVBLOX] Warning: RGB size ("
                      << rgb_undistorted.cols << "x" << rgb_undistorted.rows
                      << ") != depth size (" << depth_w << "x" << depth_h
                      << "). Color integration may be inconsistent." << std::endl;
        }

        static nvblox::ColorImage color_img(nvblox::MemoryType::kDevice);
        cvRgbToNvbloxColor(rgb_undistorted, &color_img, /*stream=*/nullptr);

        sdf_mapper_->integrateColor(color_img, T_L_C, nvb_cam);
    } else {
        std::cout << "[NVBLOX] kf.img_undist_ is empty, skipping color integration.\n";
    }
}

// torch::Tensor VoxelMapper::sampleTsdfAtPointsWorld(const torch::Tensor& pts_world)
// {
//     // pts_world: [N, 3], float32, world coordinates (CPU or CUDA)
//     TORCH_CHECK(
//         pts_world.dim() == 2 && pts_world.size(1) == 3,
//         "VoxelMapper::sampleTsdfAtPointsWorld expects pts_world of shape [N,3]");

//     // If mapper not initialized, just return zeros.
//     if (!sdf_mapper_) {
//         return torch::zeros({pts_world.size(0)}, pts_world.options());
//     }

//     // Get TSDF layer from Mapper (by reference, NOT pointer).
//     // Mapper::tsdf_layer() returns a TsdfLayer (VoxelBlockLayer<TsdfVoxel>).
//     nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();

//     // (Optional safety) If there is no TSDF data yet (no depth integrated),
//     // tsdf_layer.size() will be zero; we can early out with zeros.
//     if (tsdf_layer.size() == 0) {
//         return torch::zeros({pts_world.size(0)}, pts_world.options());
//     }

//     const bool input_on_cuda = pts_world.device().is_cuda();

//     // Work on CPU for now (N is number of voxels).
//     torch::Tensor pts_cpu = pts_world.to(torch::kCPU);
//     const int64_t N = pts_cpu.size(0);

//     // ─────────────────────────────────────────
//     // DEBUG: compare AABB of TSDF layer vs pts_world
//     // ─────────────────────────────────────────
//     {
//         static bool printed_aabb = false;
//         if (!printed_aabb) {
//             printed_aabb = true;

//             const float block_size = tsdf_layer.block_size();

//             Eigen::Vector3f tsdf_min(
//                 std::numeric_limits<float>::max(),
//                 std::numeric_limits<float>::max(),
//                 std::numeric_limits<float>::max());
//             Eigen::Vector3f tsdf_max(
//                 -std::numeric_limits<float>::max(),
//                 -std::numeric_limits<float>::max(),
//                 -std::numeric_limits<float>::max());

//             std::vector<nvblox::Index3D> block_indices = tsdf_layer.getAllBlockIndices();
//             for (const auto& idx : block_indices) {
//                 nvblox::Vector3f origin =
//                     nvblox::getPositionFromBlockIndex(block_size, idx);
//                 Eigen::Vector3f bmin(origin.x(), origin.y(), origin.z());
//                 Eigen::Vector3f bmax = bmin + Eigen::Vector3f::Constant(block_size);

//                 tsdf_min = tsdf_min.cwiseMin(bmin);
//                 tsdf_max = tsdf_max.cwiseMax(bmax);
//             }

//             Eigen::Vector3f pts_min(
//                 std::numeric_limits<float>::max(),
//                 std::numeric_limits<float>::max(),
//                 std::numeric_limits<float>::max());
//             Eigen::Vector3f pts_max(
//                 -std::numeric_limits<float>::max(),
//                 -std::numeric_limits<float>::max(),
//                 -std::numeric_limits<float>::max());

//             auto acc_pts_dbg = pts_cpu.accessor<float, 2>();
//             for (int64_t i = 0; i < N; ++i) {
//                 Eigen::Vector3f p(acc_pts_dbg[i][0],
//                                   acc_pts_dbg[i][1],
//                                   acc_pts_dbg[i][2]);
//                 pts_min = pts_min.cwiseMin(p);
//                 pts_max = pts_max.cwiseMax(p);
//             }

//             std::cout << "[TSDF SAMPLE DEBUG] TSDF layer AABB min = "
//                       << tsdf_min.transpose()
//                       << "  max = " << tsdf_max.transpose() << "\n";
//             std::cout << "[TSDF SAMPLE DEBUG] pts_world AABB   min = "
//                       << pts_min.transpose()
//                       << "  max = " << pts_max.transpose() << "\n";
//             std::cout << "[TSDF SAMPLE DEBUG] tsdf_voxel_size = "
//                       << tsdf_layer.voxel_size() << " m\n";
//         }
//     }

//     // Output SDF on CPU.
//     torch::Tensor sdf_cpu =
//         torch::empty({N}, pts_cpu.options().dtype(torch::kFloat32));

//     auto acc_pts = pts_cpu.accessor<float, 2>();
//     auto acc_sdf = sdf_cpu.accessor<float, 1>();

//     // Build vector of query positions in layer/world frame.
//     std::vector<nvblox::Vector3f> positions_L;
//     positions_L.reserve(static_cast<size_t>(N));
//     for (int64_t i = 0; i < N; ++i) {
//         positions_L.emplace_back(
//             acc_pts[i][0],
//             acc_pts[i][1],
//             acc_pts[i][2]
//         );
//     }

//     // Prepare output buffers for nvblox.
//     std::vector<nvblox::TsdfVoxel> voxels;
//     std::vector<bool> success_flags;

//     // Query TSDF voxels at those positions.
//     tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

//     // Fill SDF tensor.
//     const size_t M_vox = voxels.size();
//     const size_t M_suc = success_flags.size();

//     int64_t num_real  = 0;  // we successfully read a voxel
//     int64_t num_fake  = 0;  

//     for (int64_t i = 0; i < N; ++i) {
//         const bool have_voxel =
//             (static_cast<size_t>(i) < M_vox) &&
//             (static_cast<size_t>(i) < M_suc) &&
//             success_flags[i];
//         if (have_voxel) {
//             acc_sdf[i] = voxels[i].distance;
//             ++num_real;
//         } else {
//             acc_sdf[i] = std::numeric_limits<float>::quiet_NaN();
//             ++num_fake;
//         }
//     }
//     // Print stats about real vs fake SDFs.
//     if (N > 0) {
//         const double fake_ratio = static_cast<double>(num_fake) /
//                                   static_cast<double>(N) * 100.0;
//         const double real_ratio = static_cast<double>(num_real) /
//                                   static_cast<double>(N) * 100.0;
//         std::cout << "[TSDF SAMPLE] N=" << N
//                   << "  real=" << num_real  << " (" << real_ratio  << "%)"
//                   << "  fake=" << num_fake  << " (" << fake_ratio  << "%)"
//                   << std::endl;
//     }

//     // Move back to the original device if needed.
//     if (input_on_cuda) {
//         return sdf_cpu.to(pts_world.device());
//     } else {
//         return sdf_cpu;
//     }
// }

VoxelMapper::TsdfSample VoxelMapper::sampleTsdfAtPointsWorld(const torch::Tensor& pts_world)
{
    TORCH_CHECK(
        pts_world.defined() &&
        pts_world.dim() == 2 &&
        pts_world.size(1) == 3,
        "VoxelMapper::sampleTsdfAtPointsWorld expects pts_world of shape [N,3]");

    const auto N = pts_world.size(0);
    TsdfSample out;

    // Preserve device of input
    const bool input_on_cuda = pts_world.device().is_cuda();
    const auto out_device    = pts_world.device();

    // Default outputs (unknown)
    out.tsdf    = torch::full({N}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(out_device));
    out.weight  = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(out_device));
    out.success = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(out_device));

    // If TSDF is unavailable, return unknowns.
    if (!sdf_mapper_) {
        return out;
    }
    nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
    if (tsdf_layer.size() == 0 || N == 0) {
        return out;
    }

    // Query on CPU (nvblox API expects std::vector positions)
    torch::Tensor pts_cpu = pts_world.to(torch::kCPU).contiguous();
    auto acc_pts = pts_cpu.accessor<float, 2>();

    std::vector<nvblox::Vector3f> positions_L;
    positions_L.reserve(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) {
        positions_L.emplace_back(acc_pts[i][0], acc_pts[i][1], acc_pts[i][2]);
    }

    std::vector<nvblox::TsdfVoxel> voxels;
    std::vector<bool> success_flags;
    tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

    const size_t M_vox = voxels.size();
    const size_t M_suc = success_flags.size();

    // Fill CPU buffers first (fast, avoids per-element device writes)
    torch::Tensor tsdf_cpu   = torch::full({N}, std::numeric_limits<float>::quiet_NaN(),
                                           torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor w_cpu      = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor succ_cpu   = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    auto acc_tsdf = tsdf_cpu.accessor<float, 1>();
    auto acc_w    = w_cpu.accessor<float, 1>();
    auto acc_succ = succ_cpu.accessor<bool, 1>();

    int64_t num_success = 0;
    int64_t num_fail    = 0;

    // Weight statistics over successful reads
    float w_min = std::numeric_limits<float>::infinity();
    float w_max = 0.0f;
    double w_sum = 0.0;

    for (int64_t i = 0; i < N; ++i) {
        const bool have_voxel =
            (static_cast<size_t>(i) < M_vox) &&
            (static_cast<size_t>(i) < M_suc) &&
            success_flags[i];

        if (!have_voxel) {
            ++num_fail;
            continue;
        }

        const auto& v = voxels[i];
        acc_tsdf[i] = v.distance;
        acc_w[i]    = v.weight;
        acc_succ[i] = true;

        ++num_success;

        w_min = std::min(w_min, v.weight);
        w_max = std::max(w_max, v.weight);
        w_sum += static_cast<double>(v.weight);
    }

    // Optional: print once or occasionally
    {
        static int64_t printed = 0;
        if (printed < 5) {  // print first few calls to verify sanity
            ++printed;
            const double suc_ratio = (N > 0) ? (100.0 * double(num_success) / double(N)) : 0.0;
            const double mean_w    = (num_success > 0) ? (w_sum / double(num_success)) : 0.0;

            std::cout << "[TSDF SAMPLE] N=" << N
                      << " success=" << num_success << " (" << suc_ratio << "%)"
                      << " fail=" << num_fail
                      << " w_min=" << (num_success > 0 ? w_min : 0.0f)
                      << " w_max=" << (num_success > 0 ? w_max : 0.0f)
                      << " w_mean=" << mean_w
                      << " tsdf_voxel_size=" << tsdf_layer.voxel_size()
                      << std::endl;
        }
    }

    // Move to original device
    if (input_on_cuda) {
        out.tsdf    = tsdf_cpu.to(out_device);
        out.weight  = w_cpu.to(out_device);
        out.success = succ_cpu.to(out_device);
    } else {
        out.tsdf    = tsdf_cpu;
        out.weight  = w_cpu;
        out.success = succ_cpu;
    }

    return out;
}

static torch::Tensor computeSvrasterGridPointsWorld(
    const torch::Tensor& grid_pts_key,   // [M,3] int64
    const torch::Tensor& scene_center,   // [3] float
    const torch::Tensor& scene_extent,   // [1] float
    int max_num_levels)                  // e.g. 16
{
    TORCH_CHECK(grid_pts_key.defined() && grid_pts_key.dim() == 2 && grid_pts_key.size(1) == 3,
                "grid_pts_key must be [M,3] int64");
    TORCH_CHECK(scene_center.defined() && scene_center.numel() == 3,
                "scene_center must be [3]");
    TORCH_CHECK(scene_extent.defined() && scene_extent.numel() == 1,
                "scene_extent must be [1]");

    auto dev = scene_center.device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

    // scene_min = scene_center - 0.5 * scene_extent
    torch::Tensor scene_min = scene_center.to(opts_f).view({3}) - 0.5f * scene_extent.to(opts_f).view({1});

    // finest_vox_size = scene_extent * 2^{-MAX_NUM_LEVELS}
    // (grid_pts_key are integer coords on the finest grid; corners, not centers)
    const float scale = std::ldexp(1.0f, -max_num_levels);  // 2^{-L}
    torch::Tensor finest_vox = scene_extent.to(opts_f) * scale; // [1]

    // grid_xyz = scene_min + grid_pts_key * finest_vox_size
    torch::Tensor grid_xyz =
        scene_min.view({1,3}) + grid_pts_key.to(dev).to(torch::kFloat32) * finest_vox.view({1,1});

    return grid_xyz.contiguous(); // [M,3]
}

// Grid-point-based: gather 8 corner TSDFs for each SVRaster voxel using vox_key_
VoxelMapper::TsdfCornerSample VoxelMapper::sampleTsdfAtSvrasterGridCornersWorld()
{
    TORCH_CHECK(voxel_model_ != nullptr, "voxel_model_ is null");

    torch::Tensor grid_key = voxel_model_->gridPtsKey(); // [M,3] int64
    torch::Tensor vox_key  = voxel_model_->voxKey();     // [N,8] int64

    TORCH_CHECK(grid_key.defined() && grid_key.dim() == 2 && grid_key.size(1) == 3,
                "gridPtsKey must be [M,3]");
    TORCH_CHECK(vox_key.defined() && vox_key.dim() == 2 && vox_key.size(1) == 8,
                "voxKey must be [N,8]");

    const int64_t M = grid_key.size(0);
    const int64_t N = vox_key.size(0);

    TsdfCornerSample out;
    auto dev = voxel_model_->voxCenter().device();

    // Default (unknown)
    out.tsdf    = torch::full({N, 8}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.weight  = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.success = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kBool).device(dev));

    if (!sdf_mapper_ || sdf_mapper_->tsdf_layer().size() == 0 || N == 0 || M == 0) {
        return out;
    }

    // 1) grid_key -> world xyz
    torch::Tensor scene_center = voxel_model_->SceneCenter(); // [3]
    torch::Tensor scene_extent = voxel_model_->SceneExtent(); // [1]
    const int Lmax = voxel_model_->maxNumLevels();

    torch::Tensor grid_xyz = computeSvrasterGridPointsWorld(grid_key, scene_center, scene_extent, Lmax);
    if (grid_xyz.device() != dev) grid_xyz = grid_xyz.to(dev);

    // 2) Sample TSDF at all grid points
    TsdfSample g = sampleTsdfAtPointsWorld(grid_xyz); // g.tsdf,g.weight,g.success are [M]

    // 3) Gather per-voxel corners using vox_key
    // Flatten indices: [N,8] -> [N*8]
    torch::Tensor idx = vox_key.to(dev).to(torch::kLong).reshape({-1}); // [N*8]
    TORCH_CHECK(idx.numel() == N * 8, "vox_key reshape mismatch");

    // // Defensive bounds check (optional; can be expensive, use only while debugging)
    // TORCH_CHECK((idx >= 0).all().item<bool>() && (idx < M).all().item<bool>(),
    //             "vox_key contains out-of-range indices (must be in [0, M))");

    torch::Tensor tsdf8 = g.tsdf.index_select(0, idx).view({N, 8}).contiguous();
    torch::Tensor w8    = g.weight.index_select(0, idx).view({N, 8}).contiguous();
    torch::Tensor ok8   = g.success.index_select(0, idx).view({N, 8}).contiguous();

    out.tsdf    = tsdf8;
    out.weight  = w8;
    out.success = ok8;
    return out;
}

VoxelMapper::TsdfCornerSample VoxelMapper::sampleTsdfAtVoxelCornersWorld(
    const torch::Tensor& centers_world,
    const torch::Tensor& sizes_world)
{
    using torch::indexing::Slice;

    TORCH_CHECK(centers_world.defined() && centers_world.dim() == 2 && centers_world.size(1) == 3,
                "sampleTsdfAtVoxelCornersWorld expects centers_world [N,3]");
    TORCH_CHECK(sizes_world.defined(),
                "sampleTsdfAtVoxelCornersWorld expects sizes_world defined");

    const int64_t N = centers_world.size(0);
    TsdfCornerSample out;

    const auto dev = centers_world.device();

    // Normalize sizes to [N,1] on same device
    torch::Tensor sizes = sizes_world;
    if (sizes.dim() == 1) {
        TORCH_CHECK(sizes.size(0) == N, "sizes_world [N] mismatch with centers_world");
        sizes = sizes.view({N, 1});
    } else if (sizes.dim() == 2) {
        TORCH_CHECK(sizes.size(0) == N, "sizes_world [N,1] mismatch with centers_world");
        TORCH_CHECK(sizes.size(1) == 1, "sizes_world must have shape [N,1] if 2D");
    } else {
        TORCH_CHECK(false, "sizes_world must be [N] or [N,1]");
    }
    sizes = sizes.to(dev).contiguous();

    // Default outputs (unknown)
    out.tsdf    = torch::full({N, 8}, std::numeric_limits<float>::quiet_NaN(),
                              torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.weight  = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    out.success = torch::zeros({N, 8}, torch::TensorOptions().dtype(torch::kBool).device(dev));

    if (N == 0) return out;
    if (!sdf_mapper_) return out;
    nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
    if (tsdf_layer.size() == 0) return out;

    // Corner offsets (8,3) with all sign combinations.
    // Construct once, keep on CUDA (or whatever device centers are on).
    static torch::Tensor offsets_cache;
    static torch::Device cached_dev = torch::kCPU;

    if (!offsets_cache.defined() || cached_dev != dev) {
        // CPU literal then move
        offsets_cache = torch::tensor(
            {{-1.f, -1.f, -1.f},
             {-1.f, -1.f,  1.f},
             {-1.f,  1.f, -1.f},
             {-1.f,  1.f,  1.f},
             { 1.f, -1.f, -1.f},
             { 1.f, -1.f,  1.f},
             { 1.f,  1.f, -1.f},
             { 1.f,  1.f,  1.f}},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
        ).to(dev).contiguous();
        cached_dev = dev;
    }
    const torch::Tensor offsets = offsets_cache; // [8,3] on dev

    // Build corners: [N,8,3] = centers[:,None,:] + 0.5*sizes[:,None,:]*offsets[None,:,:]
    torch::Tensor half = 0.5f * sizes;                 // [N,1]
    torch::Tensor corners = centers_world.contiguous().view({N, 1, 3})
                          + half.view({N, 1, 1}) * offsets.view({1, 8, 3}); // [N,8,3]
    torch::Tensor corners_flat = corners.view({N * 8, 3}).contiguous();     // [N*8,3]

    // Use your existing point sampler (returns [N*8])
    TsdfSample s = sampleTsdfAtPointsWorld(corners_flat);

    // Reshape back to [N,8]
    TORCH_CHECK(s.tsdf.defined() && s.tsdf.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected tsdf size");
    TORCH_CHECK(s.weight.defined() && s.weight.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected weight size");
    TORCH_CHECK(s.success.defined() && s.success.numel() == N * 8,
                "sampleTsdfAtPointsWorld returned unexpected success size");

    out.tsdf    = s.tsdf.view({N, 8}).contiguous();
    out.weight  = s.weight.view({N, 8}).contiguous();
    out.success = s.success.view({N, 8}).contiguous();

    // ------------------------------------------------------------
    // One-time sanity check: do corners match center ± size/2 ?
    // ------------------------------------------------------------
    {
        static bool printed_once = false;
        if (!printed_once) {
            printed_once = true;

            const int64_t i0 = 0;
            if (corners.defined() && corners.dim() == 3 && corners.size(0) > i0) {
                // Bring to CPU for printing
                auto c0_cpu = centers_world.index({i0}).to(torch::kCPU).contiguous();      // [3]
                auto s0_cpu = sizes_world.index({i0}).view({1}).to(torch::kCPU).contiguous(); // [1]
                auto crn_cpu = corners.index({i0}).to(torch::kCPU).contiguous();          // [8,3]

                const float cx = c0_cpu[0].item<float>();
                const float cy = c0_cpu[1].item<float>();
                const float cz = c0_cpu[2].item<float>();
                const float s  = s0_cpu[0].item<float>();
                const float h  = 0.5f * s;

                // Expected bounds
                const float ex_min_x = cx - h, ex_max_x = cx + h;
                const float ex_min_y = cy - h, ex_max_y = cy + h;
                const float ex_min_z = cz - h, ex_max_z = cz + h;

                // Observed bounds from corners
                auto min_xyz = std::get<0>(crn_cpu.min(/*dim=*/0, /*keepdim=*/false)); // [3]
                auto max_xyz = std::get<0>(crn_cpu.max(/*dim=*/0, /*keepdim=*/false)); // [3]

                const float ob_min_x = min_xyz[0].item<float>();
                const float ob_min_y = min_xyz[1].item<float>();
                const float ob_min_z = min_xyz[2].item<float>();

                const float ob_max_x = max_xyz[0].item<float>();
                const float ob_max_y = max_xyz[1].item<float>();
                const float ob_max_z = max_xyz[2].item<float>();

                auto absf = [](float v){ return v < 0.f ? -v : v; };
                const float eps = 1e-5f;

                std::cout << "[TSDF CORNER SANITY] i0=" << i0 << "\n";
                std::cout << "  center = [" << cx << ", " << cy << ", " << cz << "]\n";
                std::cout << "  size   = " << s << "  half=" << h << "\n";

                std::cout << "  expected min = [" << ex_min_x << ", " << ex_min_y << ", " << ex_min_z << "]\n";
                std::cout << "  expected max = [" << ex_max_x << ", " << ex_max_y << ", " << ex_max_z << "]\n";

                std::cout << "  observed min = [" << ob_min_x << ", " << ob_min_y << ", " << ob_min_z << "]\n";
                std::cout << "  observed max = [" << ob_max_x << ", " << ob_max_y << ", " << ob_max_z << "]\n";

                const bool ok_x = (absf(ob_min_x - ex_min_x) < eps) && (absf(ob_max_x - ex_max_x) < eps);
                const bool ok_y = (absf(ob_min_y - ex_min_y) < eps) && (absf(ob_max_y - ex_max_y) < eps);
                const bool ok_z = (absf(ob_min_z - ex_min_z) < eps) && (absf(ob_max_z - ex_max_z) < eps);

                std::cout << "  axis check: x=" << (ok_x ? "OK" : "FAIL")
                        << " y=" << (ok_y ? "OK" : "FAIL")
                        << " z=" << (ok_z ? "OK" : "FAIL")
                        << " (eps=" << eps << ")\n";

                std::cout << "  corners[8,3] =\n" << crn_cpu << "\n";
            } else {
                std::cout << "[TSDF CORNER SANITY] corners tensor not ready or empty.\n";
            }
        }
    }

    return out;
}

void VoxelMapper::run()
{
    /* expose our helper scripts to the embedded Python side */
    py::gil_scoped_acquire gil;
    py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");

    // Initialize Rerun in "headless" mode (no viewer window).
    sv::RerunVisualizerBridge::instance().init(
        "PhotoSLAM-SVRaster",
        /*spawn_viewer=*/false
    );

    if (use_tsdf_mapping_) 
    {
        initializeNvbloxMapper();
    }
    // First loop: Initial gaussian mapping
    while (!isStopped())
    {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions())
        {
            mpSLAM->getAtlas()->clearMappingOperation();

            // Pull sparse SLAM map (get keyframes and map points)
            auto pMap = mpSLAM->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vKFs;
            std::vector<ORB_SLAM3::MapPoint*> vMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vKFs = pMap->GetAllKeyFrames();
                vMPs = pMap->GetAllMapPoints();
                for (const auto& pMP : vMPs)
                {
                     Point3D point3D;
                     auto pos = pMP->GetWorldPos();
                     point3D.xyz_(0) = pos.x();
                     point3D.xyz_(1) = pos.y();
                     point3D.xyz_(2) = pos.z();
                     auto color = pMP->GetColorRGB();
                     point3D.color_(0) = color(0);
                     point3D.color_(1) = color(1);
                     point3D.color_(2) = color(2);
                     scene_->cachePoint3D(pMP->mnId, point3D);
                 }
                // B) Create VoxelKeyframes from each SLAM KeyFrame
                for (const auto& pKF : vKFs)
                {
                    std::shared_ptr<VoxelKeyframe> new_kf = std::make_shared<VoxelKeyframe>(pKF->mnId, getIteration());
                    new_kf->znear_ = z_near_;
                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>()
                    );
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    // Debug: Print Keyframe pose
                    // std::cout << "[DEBUG] Keyframe ID: " << pKF->mnId 
                    //           << " Pose: (" << pose.translation().x() << ", " << pose.translation().y() << ", " << pose.translation().z() << ")\n";
                    // Camera
                    sv::Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                    new_kf->setCameraParams(camera);

                    // Image (left if STEREO)
                    cv::Mat imgRGB = pKF->imgLeftRGB;
                    // camera.undistortImage(imgRGB, imgRGB_undistorted);
                    if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                    // Auxiliary Image
                    cv::Mat imgAux = pKF->imgAuxiliary;
                    if (this->sensor_type_ == RGBD)
                        camera.undistortImage(imgAux, imgAux_undistorted);
                    else
                        imgAux_undistorted = imgAux;

                    // {
                    //     static const auto proj_dir = result_dir_ / "proj_debug";
                    //     static const auto imgs_dir = proj_dir / "imgs";
                    //     std::filesystem::create_directories(imgs_dir);
                    //     // imgRGB_undistorted is CV_32FC3 RGB in [0..1] (in Photo-SLAM). If it’s 8U, convert first:
                    //     cv::Mat img_float;
                    //     if (imgRGB_undistorted.type() == CV_32FC3) {
                    //         img_float = imgRGB_undistorted;
                    //     } else {
                    //         imgRGB_undistorted.convertTo(img_float, CV_32FC3, 1.0/255.0);
                    //     }
                    //     saveKfPng_fromFloatRGB(img_float, pKF->mnId, imgs_dir);
                    // }

                    new_kf->original_image_ =
                        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                    // std::cout << "new_kf->original_image_" << new_kf->original_image_ << std::endl;
                    new_kf->img_filename_ = pKF->mNameFile;
                    new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                    new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                    new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;

                    // Compute transformations
                    // new_kf->computeTransformTensors(); //useless
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                    // // Features for increasePcdByKeyframeInactiveGeoDensify
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;

                    // ── Log this initial keyframe to Rerun as a camera ──
                    try {
                        const unsigned long kf_id = pKF->mnId;

                        // Use full resolution of this keyframe
                        const int image_height = new_kf->image_height_;
                        const int image_width  = new_kf->image_width_;

                        // Build MiniCam (this uses Tcw_ internally and sets c2w/w2c)
                        sv::MiniCam cam = new_kf->toMiniCam(image_height, image_width);

                        // Intrinsics for Rerun
                        const float fx = static_cast<float>(camera.fx());
                        const float fy = static_cast<float>(camera.fy());
                        const float cx = static_cast<float>(camera.cx());
                        const float cy = static_cast<float>(camera.cy());

                        // cam.c2w is a 4x4 torch::Tensor on CPU or CUDA; move to CPU & map to Eigen
                        torch::Tensor c2w_cpu = cam.c2w.to(torch::kCPU).contiguous();
                        TORCH_CHECK(c2w_cpu.sizes() == torch::IntArrayRef({4, 4}),
                                    "MiniCam.c2w must be 4x4");

                        Eigen::Matrix4f T_W_C;
                        {
                            float* data = c2w_cpu.data_ptr<float>();
                            Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>
                                T_row_major(data);
                            T_W_C = T_row_major;
                        }

                        // Use the undistorted RGB image as tracking image
                        const cv::Mat& track_img = new_kf->img_undist_;

                        sv::RerunVisualizerBridge::instance().visualizeCamera(
                            T_W_C,
                            track_img,
                            std::vector<Eigen::Vector2f>{},  // no 2D keypoints for now
                            std::vector<int>{},              // no track ids
                            static_cast<int>(kf_id),
                            fx, fy, cx, cy
                        );
                    } catch (const c10::Error& e) {
                        std::cerr << "[RERUN] Torch error in visualizeCamera (initial KFs): "
                                << e.msg() << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "[RERUN] Exception in visualizeCamera (initial KFs): "
                                << e.what() << std::endl;
                    }

                    // ─── nvblox: integrate this keyframe’s depth into TSDF ───
                    if (sensor_type_ == RGBD && use_tsdf_mapping_) {
                        // Assume imgAux_undistorted is a depth map aligned to imgRGB_undistorted
                        cv::Mat depth_meters;
                        if (imgAux_undistorted.type() == CV_32FC1) {
                            // already in meters
                            depth_meters = imgAux_undistorted;
                        } else if (imgAux_undistorted.type() == CV_16UC1) {
                            // common RealSense-style millimeters → meters conversion
                            imgAux_undistorted.convertTo(depth_meters, CV_32FC1, 1.0 / 1000.0);
                        } else {
                            // fallback: convert to float, assume already in meters scale
                            imgAux_undistorted.convertTo(depth_meters, CV_32FC1);
                        }
                        integrateKeyframeIntoNvblox(*new_kf, depth_meters);
                    }

                }
            }   // Mutex released

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                        // std::cout << "pkf->gaus_pyramid_original_image_[l]" << pkf->gaus_pyramid_original_image_[l] << std::endl;
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }

            // aabb_min_.setConstant( std::numeric_limits<float>::infinity());
            // aabb_max_.setConstant(-std::numeric_limits<float>::infinity());
            // have_bounds_ = false;
            // for (const auto& kv : scene_->cached_point_cloud_) {
            //     // cast<float>() returns a temporary → store as a VALUE, not a const&
            //     Eigen::Vector3f P = kv.second.xyz_.cast<float>();
            //     extendAABB(aabb_min_, aabb_max_, P);
            //     have_bounds_ = true;
            // }
            // if (have_bounds_) {
            //     std::cout.setf(std::ios::fixed);
            //     // std::cout << std::setprecision(6)
            //     //         << "[AABB:init] min:[" << aabb_min_.x() << "," << aabb_min_.y() << "," << aabb_min_.z()
            //     //         << "] max:[" << aabb_max_.x() << "," << aabb_max_.y() << "," << aabb_max_.z() << "]\n";
            // }
            // dumpKeyframesForProjectionFile(
            //     scene_->keyframes(),        // all KFs currently in the scene
            //     scene_->cameras_,           // intrinsics you already use
            //     result_dir_ / "proj_debug"  // output folder
            // );

            // C) Create MiniCams for all keyframes and use them for densification later
            std::vector<sv::MiniCam> tr_cams;
            tr_cams.reserve(scene_->keyframes().size());
            for (auto& kv : scene_->keyframes()) {
                // if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
                auto& kf = *kv.second;
                // Use full-res here; you can choose a smaller level if you like.
                tr_cams.emplace_back(kf.toMiniCam(kf.image_height_, kf.image_width_));
            }
            if (!tr_cams.empty()) {
                const auto& c0 = tr_cams.front();

                // position and lookat are 1D tensors of size 3 on CPU
                auto px = c0.position.index({0}).item<float>();
                auto py = c0.position.index({1}).item<float>();
                auto pz = c0.position.index({2}).item<float>();

                auto lx = c0.lookat.index({0}).item<float>();
                auto ly = c0.lookat.index({1}).item<float>();
                auto lz = c0.lookat.index({2}).item<float>();

                std::cout << "[MiniCam debug] first cam: "
                        << "pos=(" << px << "," << py << "," << pz << ") "
                        << "lookat=(" << lx << "," << ly << "," << lz << ") "
                        << "pix_size=" << c0.pix_size
                        << std::endl;
            }
            // D) Create voxel model & trainer setup
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                // scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                // std::cout << "[VoxelMapper] Scene extent: " 
                //             << scene_->cameras_extent_ << std::endl;
                // save_initial_pcd_npy(result_dir_, scene_->cached_point_cloud_);
                // auto restored = load_full_pcd_from_logs((result_dir_ / "offline_experiment").string());
                // voxel_model_->createFromPcd(std::move(restored));
                voxel_model_->createFromPcd(scene_->cached_point_cloud_, tr_cams);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                voxel_model_->createTrainer(
                                            opt_params_.geo_lr_,
                                            opt_params_.sh0_lr_,
                                            opt_params_.shs_lr_,
                                            opt_params_.optim_beta1_,
                                            opt_params_.optim_beta2_,
                                            opt_params_.optim_eps_,
                                            opt_params_.lr_decay_ckpt_,
                                            opt_params_.lr_decay_mult_);
            }

            // One warm-up optimization step
            trainForOneIteration();

            initial_mapped_ = true;
            break;  // Exit the initial mapping loop
        }
        else if (mpSLAM->isShutDown())
        {
            break;
        }
        else
        {
            // Initial conditions not satisfied yet
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental voxel mapping
     int SLAM_stop_iter = 0;
     while (!isStopped()) {
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_)
                cullKeyframes();
        }
 
        // Invoke training once
        trainForOneIteration();

        if (mpSLAM->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }

    // // Third loop: Tail gaussian optimization
    int adapt_interval = opt_params_.adapt_every_;          // cfg.procedure.adapt_every
    int n_delay_iters  = adapt_interval * 0.8f;        // same heuristic as GS code
    while (getIteration() - SLAM_stop_iter <= n_delay_iters
        || (getIteration() % adapt_interval) <= n_delay_iters
        || isKeepingTraining() )
    {
        trainForOneIteration();
        // Re-read in case user changed cfg at runtime
        adapt_interval = opt_params_.adapt_every_;
        n_delay_iters  = adapt_interval * 0.8f;
    }
    // {
    //     // 1) Check if we had a recent artifact fill (fill_empty_cells_)
    //     bool need_forced_densify = false;
    //     if (last_artifact_fill_iter_ >= 0 && last_densify_iter_ >= 0) {
    //         if (last_artifact_fill_iter_ > last_densify_iter_) {
    //             need_forced_densify = true;
    //             std::cout << "[VoxelMapper] Tail: artifact fill after last densify "
    //                     << "(last_fill_iter=" << last_artifact_fill_iter_
    //                     << ", last_densify_iter=" << last_densify_iter_
    //                     << ") -> will force one densification.\n";
    //         }
    //     }
    //     // 2) If needed, run ONE forced densification step (prune+subdivide)
    //     if (need_forced_densify) {
    //         // Backup optimization parameters and last_artifact_fill_iter_
    //         auto   backup_opt        = opt_params_;
    //         auto   backup_last_fill  = last_artifact_fill_iter_;
    //         const int cur_iter_before = getIteration();

    //         // Configure opt_params_ so that the NEXT iteration always densifies:
    //         //   - adapt_every_ = 1 → meet_adapt_period every iteration
    //         //   - adapt_from_ ≤ current_iter
    //         //   - prune_until_ / subdivide_until_ ≥ current_iter+1
    //         opt_params_.adapt_every_     = 1;
    //         opt_params_.adapt_from_      =
    //             std::min(backup_opt.adapt_from_, cur_iter_before);
    //         opt_params_.prune_until_     =
    //             std::max(backup_opt.prune_until_, cur_iter_before + 1);
    //         opt_params_.subdivide_until_ =
    //             std::max(backup_opt.subdivide_until_, cur_iter_before + 1);

    //         // Disable cooldown for this forced step so densification actually runs
    //         last_artifact_fill_iter_ = -1;

    //         // This call WILL:
    //         //  - do one training iteration
    //         //  - trigger prune/subdivide according to the modified opt_params_
    //         trainForOneIteration();

    //         // Restore original parameters and state
    //         opt_params_             = backup_opt;
    //         last_artifact_fill_iter_ = backup_last_fill;

    //         std::cout << "[VoxelMapper] Tail: forced densification step done at iter "
    //                 << getIteration() << "\n";
    //     }
    //     // 3) Tail optimization: run 300 extra iterations AFTER densification (if any)
    //     const int tail_extra_iters = 300;  // can move to YAML later
    //     const int start_tail_iter  = getIteration();
    //     const int end_tail_iter    = start_tail_iter + tail_extra_iters;
    //     while (getIteration() < end_tail_iter) {
    //         trainForOneIteration();
    //     }
    // }

    // ─────────────────────────────────────────────────────────────
    // TSDF-based transparency AFTER all training is done
    // ─────────────────────────────────────────────────────────────
    // if (sensor_type_ == RGBD && use_tsdf_mapping_) {
    //     applyFinalTsdfTransparency();
    // }

    // if (have_bounds_) {
    //     std::cout.setf(std::ios::fixed);
    //     std::cout << std::setprecision(6)
    //             << "[AABB:final] min:[" << aabb_min_.x() << "," << aabb_min_.y() << "," << aabb_min_.z()
    //             << "] max:[" << aabb_max_.x() << "," << aabb_max_.y() << "," << aabb_max_.z() << "]\n";
    // }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    const auto rrd_dir  = result_dir_ / "rerun";
    std::filesystem::create_directories(rrd_dir);
    const auto rrd_path = (rrd_dir / "recording.rrd").string();
    sv::RerunVisualizerBridge::instance().saveRecording(rrd_path);

    signalStop();
 }

 // ---------- debug helpers ----------
static std::string shp(const torch::Tensor& t) {
    if (!t.defined()) return "<undef>";
    std::ostringstream oss; oss << '[';
    for (int i=0;i<t.dim();++i){ if(i) oss<<','; oss<<t.size(i); }
    oss << ']'; return oss.str();
}
static std::string dev(const torch::Tensor& t) {
    return t.defined() ? t.device().str() : "<undef>";
}
static void dump_param(const char* name, const torch::Tensor& p) {
    std::cout << "    " << name
              << " def=" << (p.defined()?1:0)
              << " req=" << (p.defined()?p.requires_grad():false)
              << " shape=" << shp(p)
              << " dtype=" << (p.defined()?c10::toString(p.scalar_type()):"<undef>")
              << " dev=" << dev(p);
    torch::Tensor g;
    try { if (p.defined()) g = p.grad(); } catch(...) {}
    std::cout << "  | grad def=" << (g.defined()?1:0)
              << " shape=" << shp(g) << "\n";
}

void VoxelMapper::savePhotometricErrorHeatmapAsPng(
    const torch::Tensor& error_tensor,    // [H,W] or [1,H,W] or [3,H,W]
    const std::filesystem::path& path)
{
    if (!error_tensor.defined()) {
        std::cout << "[PHOTO-ERR] tensor is undefined, skip save\n";
        return;
    }

    torch::Tensor e = error_tensor.detach()
                                     .to(torch::kCPU)
                                     .to(torch::kFloat32);

    // Handle shapes [C,H,W] → [H,W]
    if (e.dim() == 3) {
        const auto C = e.size(0);
        if (C == 1) {
            e = e.squeeze(0);          // [1,H,W] -> [H,W]
        } else if (C >= 3) {
            // Average over channels to get scalar error
            e = e.mean(0);             // [3,H,W] -> [H,W]
        } else {
            std::cout << "[PHOTO-ERR] unexpected 3D shape: "
                      << e.sizes() << ", skip save\n";
            return;
        }
    } else if (e.dim() != 2) {
        std::cout << "[PHOTO-ERR] expected [H,W] or [C,H,W], got dim="
                  << e.dim() << ", skip save\n";
        return;
    }

    e = e.contiguous();  // [H,W]

    // Mask valid pixels (error > 0); you can make this looser/tighter
    torch::Tensor mask = e > 0.0f;
    if (!mask.any().item<bool>()) {
        std::cout << "[PHOTO-ERR] all errors <= 0, skip save\n";
        return;
    }

    // Use quantiles to clip outliers (3%–97%), similar to depth viz
    torch::Tensor e_valid = e.masked_select(mask);
    if (e_valid.numel() < 10) {
        std::cout << "[PHOTO-ERR] too few valid pixels, skip save\n";
        return;
    }

    torch::Tensor q = torch::quantile(
        e_valid,
        torch::tensor({0.03f, 0.97f}, torch::kFloat32)
    );
    float e_min = q[0].item<float>();
    float e_max = q[1].item<float>();

    if (!(e_max > e_min)) {
        std::cout << "[PHOTO-ERR] invalid quantile range: min=" << e_min
                  << " max=" << e_max << ", skip save\n";
        return;
    }

    // Normalize to [0,1]
    torch::Tensor x = (e - e_min) / (e_max - e_min);
    x = x.clamp(0.0f, 1.0f);

    // Map to [0,255]
    torch::Tensor x_u8 = (x * 255.0f).to(torch::kUInt8).contiguous();
    int H = static_cast<int>(x_u8.size(0));
    int W = static_cast<int>(x_u8.size(1));

    cv::Mat gray(H, W, CV_8UC1, x_u8.data_ptr<uint8_t>());

    // Colorize with VIRIDIS
    cv::Mat color_bgr;
    cv::applyColorMap(gray, color_bgr, cv::COLORMAP_VIRIDIS);

    // Zero-out invalid pixels (where mask==0)
    torch::Tensor mask_u8 = mask.to(torch::kUInt8).contiguous();
    cv::Mat mask_cv(H, W, CV_8UC1, mask_u8.data_ptr<uint8_t>());
    color_bgr.setTo(cv::Scalar(0, 0, 0), mask_cv == 0);

    std::filesystem::create_directories(path.parent_path());
    cv::imwrite(path.string(), color_bgr);
    // if (!cv::imwrite(path.string(), color_bgr)) {
    //     std::cerr << "[PHOTO-ERR] failed to write PNG: " << path << "\n";
    // } else {
    //     std::cout << "[PHOTO-ERR] wrote heatmap PNG: " << path << "\n";
    // }
}

void VoxelMapper::saveDepthTensorAsPng(
    const torch::Tensor& depth_tensor,
    const std::filesystem::path& path)
{
    if (!depth_tensor.defined()) {
        std::cout << "[DEPTH VIZ] tensor is undefined, skip save\n";
        return;
    }

    // Move to CPU, float
    torch::Tensor d = depth_tensor.detach()
                                   .to(torch::kCPU)
                                   .to(torch::kFloat32);

    // Handle shapes [1,H,W] or [3,H,W] like before
    if (d.dim() == 3) {
        const auto C = d.size(0);
        if (C == 1) {
            d = d.squeeze(0);  // [1,H,W] -> [H,W]
        } else if (C >= 3) {
            // std::cout << "[DEPTH VIZ] got [3,H,W], using channel 2 as d_med\n";
            d = d.index({2});  // [3,H,W] -> [H,W] (median depth)
        } else {
            // std::cout << "[DEPTH VIZ] unexpected 3D shape: "
            //           << d.sizes() << ", skip save\n";
            return;
        }
    } else if (d.dim() != 2) {
        // std::cout << "[DEPTH VIZ] expected [H,W] or [1,H,W] or [3,H,W], got dim="
        //           << d.dim() << ", skip save\n";
        return;
    }

    d = d.contiguous(); // [H,W]

    // --- SVRaster-style mask: valid where depth > 0 ---
    torch::Tensor mask = d > 0.0f;
    if (!mask.any().item<bool>()) {
        std::cout << "[DEPTH VIZ] all depth <= 0, skip save\n";
        return;
    }

    // Take only valid values for quantiles
    torch::Tensor d_valid = d.masked_select(mask);
    if (d_valid.numel() < 10) {
        std::cout << "[DEPTH VIZ] too few valid depths, skip save\n";
        return;
    }

    // --- Quantiles 3% and 97% (like np.quantile) ---
    torch::Tensor q = torch::quantile(
        d_valid,
        torch::tensor({0.03f, 0.97f}, torch::TensorOptions().dtype(torch::kFloat32))
    );
    float d_min = q[0].item<float>();
    float d_max = q[1].item<float>();

    if (!(d_max > d_min)) {
        std::cout << "[DEPTH VIZ] invalid quantile range: min=" << d_min
                  << " max=" << d_max << ", skip save\n";
        return;
    }

    // --- Log remapping as in viz_tensordepth_log ---
    torch::Tensor x = d - d_min;              // x - dmin
    x = x + 1.0f;                             // 1 + x - dmin
    x = x.clamp(1.0f, 1e9f);                  // clip(1, 1e9)
    x = x.log();                              // log()

    float denom = std::log(1.0f + (d_max - d_min)); // log(1 + dmax-dmin)
    if (!(denom > 0.0f)) {
        denom = 1.0f;
    }
    x = x / denom;                            // normalize
    x = x.clamp(0.0f, 1.0f);

    // to [0,255] uint8
    torch::Tensor x_u8 = (x * 255.0f).to(torch::kUInt8).contiguous();

    int H = static_cast<int>(x_u8.size(0));
    int W = static_cast<int>(x_u8.size(1));

    // Gray image
    cv::Mat gray(H, W, CV_8UC1, x_u8.data_ptr<uint8_t>());

    // Colorize with VIRIDIS (same as cv2.COLORMAP_VIRIDIS)
    cv::Mat color_bgr;
    cv::applyColorMap(gray, color_bgr, cv::COLORMAP_VIRIDIS);

    // Zero-out invalid pixels (mask == 0) like viz[~m] = 0
    torch::Tensor mask_u8 = mask.to(torch::kUInt8).contiguous();
    cv::Mat mask_cv(H, W, CV_8UC1, mask_u8.data_ptr<uint8_t>());
    color_bgr.setTo(cv::Scalar(0, 0, 0), mask_cv == 0);

    // (Optional) convert BGR->RGB. For PNG on disk it usually does not matter,
    // but if you want strict RGB ordering:
    // cv::Mat color_rgb;
    // cv::cvtColor(color_bgr, color_rgb, cv::COLOR_BGR2RGB);
    // auto& final_img = color_rgb;
    cv::Mat& final_img = color_bgr;

    std::filesystem::create_directories(path.parent_path());
    cv::imwrite(path.string(), final_img);
    // if (!cv::imwrite(path.string(), final_img)) {
    //     std::cerr << "[DEPTH VIZ] failed to write depth PNG to: " 
    //               << path << "\n";
    // } else {
    //     std::cout << "[DEPTH VIZ] wrote depth PNG: " << path << "\n";
    // }
}

bool VoxelMapper::buildSparseDepthFromMapPoints(
    const sv::MiniCam& cam,
    int image_width,
    int image_height,
    torch::Tensor& sparse_uv,
    torch::Tensor& sparse_depth)
{
    // 0) Pull global SLAM point cloud (world coords) from the scene
    const auto& pcd = scene_->cached_point_cloud_;
    const int64_t M_total = static_cast<int64_t>(pcd.size());
    if (M_total == 0) {
        return false;
    }

    // 1) Pack world points into a host vector [M_total, 3]
    std::vector<float> host_pts;
    host_pts.reserve(3 * M_total);
    for (const auto& kv : pcd) {
        const Point3D& P = kv.second;          // you already fill xyz_ in run()
        host_pts.push_back(static_cast<float>(P.xyz_(0)));
        host_pts.push_back(static_cast<float>(P.xyz_(1)));
        host_pts.push_back(static_cast<float>(P.xyz_(2)));
    }

    auto opts_host = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor pts_world_cpu = torch::from_blob(
        host_pts.data(),
        {M_total, 3},
        opts_host);

    // Move to device and own the memory (clone())
    auto opts_dev = torch::TensorOptions().dtype(torch::kFloat32).device(mDevice);
    torch::Tensor pts_world = pts_world_cpu.clone().to(mDevice);   // [M,3]

    // 2) Transform world → camera using cam.w2c  (SVRaster-style)
    //
    // We build homogeneous coordinates [M,4] and multiply by w2c^T:
    //   X_cam = X_world_h @ w2c^T
    //
    torch::Tensor ones = torch::ones({M_total, 1}, opts_dev);
    torch::Tensor pts_world_h = torch::cat({pts_world, ones}, /*dim=*/1); // [M,4]

    torch::Tensor w2c = cam.w2c.to(mDevice);                                // [4,4]
    torch::Tensor pts_cam_h =
        torch::matmul(pts_world_h, w2c.transpose(0, 1));                    // [M,4]
    torch::Tensor pts_cam = pts_cam_h.index(
        {torch::indexing::Slice(), torch::indexing::Slice(0, 3)});          // [M,3]

    torch::Tensor X = pts_cam.index({torch::indexing::Slice(), 0}); // [M]
    torch::Tensor Y = pts_cam.index({torch::indexing::Slice(), 1}); // [M]
    torch::Tensor Z = pts_cam.index({torch::indexing::Slice(), 2}); // [M]

    // 3) Compute intrinsics from tanFOV + cx,cy (exactly what rasterizer uses)
    const float W = static_cast<float>(image_width);
    const float H = static_cast<float>(image_height);

    const float fx = 0.5f * W / cam.tanfovx;
    const float fy = 0.5f * H / cam.tanfovy;

    // u,v in pixel coords
    torch::Tensor u = fx * X / Z + cam.cx;   // [M]
    torch::Tensor v = fy * Y / Z + cam.cy;   // [M]

    // 4) Visibility & image bounds
    torch::Tensor valid =
        (Z > 0.0f) &
        (u >= 0.0f) & (u <= (W - 1.0f)) &
        (v >= 0.0f) & (v <= (H - 1.0f));     // [M]

    torch::Tensor valid_idx = torch::nonzero(valid).squeeze(1); // [M_vis]
    const int64_t M_vis = valid_idx.size(0);
    if (M_vis == 0) {
        return false;
    }

    // 5) Subsample to at most N_max points (same spirit as RGB-D version)
    const int64_t N_max = 3000;
    torch::Tensor chosen_idx;
    if (M_vis <= N_max) {
        chosen_idx = valid_idx;
    } else {
        const int64_t stride = std::max<int64_t>(int64_t(1), M_vis / N_max);
        torch::Tensor arange_idx = torch::arange(
            0, M_vis, stride,
            torch::TensorOptions().dtype(torch::kLong).device(valid_idx.device()));
        if (arange_idx.size(0) > N_max) {
            arange_idx = arange_idx.slice(0, 0, N_max);
        }
        chosen_idx = valid_idx.index_select(0, arange_idx); // [N]
    }

    const int64_t N = chosen_idx.size(0);
    if (N == 0) {
        return false;
    }

    // 6) Gather u, v, Z for the chosen points
    torch::Tensor u_chosen = u.index_select(0, chosen_idx); // [N]
    torch::Tensor v_chosen = v.index_select(0, chosen_idx); // [N]
    torch::Tensor z_chosen = Z.index_select(0, chosen_idx); // [N]

    // 7) Convert pixel coords to NDC in [-1,1] for grid_sample
    torch::Tensor u_ndc =
        2.0f * (u_chosen / (W - 1.0f)) - 1.0f;              // [N]
    torch::Tensor v_ndc =
        2.0f * (v_chosen / (H - 1.0f)) - 1.0f;              // [N]

    sparse_uv    = torch::stack({u_ndc, v_ndc}, /*dim=*/1); // [N,2]
    sparse_depth = z_chosen;                                // [N]

    return true;
}

torch::Tensor VoxelMapper::computeSparseDepthLoss_Points(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const sv::MiniCam& cam,
    int image_width,
    int image_height,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    // 0) Weight or schedule off -> no contribution
    if (opt_params_.lambda_sparse_depth_ <= 0.0f)
        return zero;

    loss_utils::SparseDepthLoss sparse_depth_loss(opt_params_.sparse_depth_until_);
    if (!sparse_depth_loss.isActive(iteration))
        return zero;

    // 1) Get raw_T / raw_depth (SVRaster-style)
    auto it_T = render_pkg.find("raw_T");
    if (it_T == render_pkg.end())
        it_T = render_pkg.find("T");          // fallback

    auto it_depth = render_pkg.find("raw_depth");
    if (it_depth == render_pkg.end())
        it_depth = render_pkg.find("depth");  // fallback

    if (it_T == render_pkg.end() || it_depth == render_pkg.end()) {
        return zero;
    }

    torch::Tensor raw_T     = it_T->second.to(mDevice);
    torch::Tensor raw_depth = it_depth->second.to(mDevice);

    // 2) Build (sparse_uv, sparse_depth) from SLAM 3D points (SVRaster-style)
    torch::Tensor sparse_uv;     // [N,2]
    torch::Tensor sparse_depth;  // [N]
    if (!buildSparseDepthFromMapPoints(cam, image_width, image_height,
                                       sparse_uv, sparse_depth)) {
        // No visible 3D points for this viewpoint
        return zero;
    }

    // 3) Low-level SparseDepthLoss (exact math as SVRaster’s __call__)
    torch::Tensor depth_loss =
        sparse_depth_loss(raw_T, raw_depth, sparse_uv, sparse_depth);

    return depth_loss;
}

bool VoxelMapper::buildSparseDepthFromRGBD(
    const std::shared_ptr<VoxelKeyframe>& kf,
    torch::Tensor& sparse_uv,
    torch::Tensor& sparse_depth)
{
    // 1) Check that this keyframe actually has RGB-D depth
    if (kf->img_auxiliary_undist_.empty()) {
        // std::cout << "[SparseDepth] kf fid=" << kf->fid_
        //           << " has EMPTY img_auxiliary_undist_ (no RGB-D)\n";
        return false;
    }

    const int H = kf->image_height_;
    const int W = kf->image_width_;

    // 2) Upload cv::Mat depth to GPU and convert to torch::Tensor
    cv::cuda::GpuMat depth_gpu;
    depth_gpu.upload(kf->img_auxiliary_undist_);

    torch::Tensor depth =
        tensor_utils::cvGpuMat2TorchTensor_Float32(depth_gpu).to(mDevice);

    // Expected shapes: [H,W] or [1,H,W]
    if (depth.dim() == 3 && depth.size(0) == 1) {
        depth = depth.squeeze(0);        // [H,W]
    }
    else if (depth.dim() != 2) {
        // std::cerr << "[SparseDepth] Unexpected depth tensor shape for fid="
        //           << kf->fid_ << " : " << depth.sizes() << std::endl;
        return false;
    }

    // Debug: print depth stats
    auto dmin = depth.min().item<float>();
    auto dmax = depth.max().item<float>();
    // std::cout << "[SparseDepth] fid=" << kf->fid_
    //           << " depth min=" << dmin
    //           << " max=" << dmax << std::endl;

    // 3) Build a validity mask based on depth range
    //    (reuse Photo-SLAM's RGBD_min_depth_ / RGBD_max_depth_ thresholds)
    //    You may want to relax RGBD_min_depth_ to >0 first for debugging.
    torch::Tensor valid_mask =
        (depth > RGBD_min_depth_) & (depth < RGBD_max_depth_);   // [H,W]

    torch::Tensor flat_mask = valid_mask.view({-1});             // [H*W]
    torch::Tensor valid_idx = torch::nonzero(flat_mask).squeeze(1); // [M]

    const auto M = valid_idx.size(0);
    // std::cout << "[SparseDepth] fid=" << kf->fid_
    //           << " valid M=" << M << " (H*W=" << (H*W) << ")\n";

    if (M == 0) {
        // No valid depth pixels in this frame
        return false;
    }

    // 4) Subsample to at most N_max points to keep training cost reasonable
    const int64_t N_max = 3000;
    torch::Tensor chosen_idx;
    if (M <= N_max) {
        chosen_idx = valid_idx;
    } else {
        const int64_t stride = std::max<int64_t>(1, M / N_max);
        torch::Tensor arange_idx = torch::arange(
            0, M, stride,
            torch::TensorOptions().dtype(torch::kLong).device(valid_idx.device()));
        if (arange_idx.size(0) > N_max) {
            arange_idx = arange_idx.slice(0, 0, N_max);
        }
        chosen_idx = valid_idx.index_select(0, arange_idx);  // [N <= N_max]
    }

    const auto N = chosen_idx.size(0);
    if (N == 0) {
        // std::cout << "[SparseDepth] fid=" << kf->fid_
        //           << " selected N=0 after subsampling\n";
        return false;
    }

    // 5) Convert flat indices to (u,v) pixel coordinates
    torch::Tensor idx_u = chosen_idx.remainder(W);   // [N]
    torch::Tensor idx_v = chosen_idx / W;           // [N]

    // 6) Convert (u,v) pixels to NDC coordinates in [-1, 1] for grid_sample
    torch::Tensor u_ndc =
        2.0f * (idx_u.to(torch::kFloat32) / float(W - 1)) - 1.0f;
    torch::Tensor v_ndc =
        2.0f * (idx_v.to(torch::kFloat32) / float(H - 1)) - 1.0f;

    sparse_uv = torch::stack({u_ndc, v_ndc}, /*dim=*/1);  // [N,2]

    // 7) Gather depths at those indices
    torch::Tensor depth_flat = depth.view({H * W});        // [H*W]
    sparse_depth = depth_flat.index_select(0, chosen_idx); // [N]

    sparse_uv    = sparse_uv.to(mDevice);
    sparse_depth = sparse_depth.to(mDevice);

    // std::cout << "[SparseDepth] fid=" << kf->fid_
    //           << " final N=" << N << " depth_sparse min="
    //           << sparse_depth.min().item<float>()
    //           << " max=" << sparse_depth.max().item<float>()
    //           << std::endl;

    return true;
}

void VoxelMapper::debugDepthStats(const cv::Mat& depth_meters, int kf_id)
{
    if (depth_meters.empty() || depth_meters.type() != CV_32FC1) {
        // std::cout << "[DEPTH DEBUG] KF " << kf_id
        //           << " depth empty or wrong type: " << depth_meters.type() << "\n";
        return;
    }

    double min_val, max_val;
    cv::minMaxLoc(depth_meters, &min_val, &max_val);

    // Count valid depths ( >0 ) and zeros
    int valid_count = 0;
    int zero_count  = 0;
    double sum_valid = 0.0;

    const int H = depth_meters.rows;
    const int W = depth_meters.cols;

    for (int v = 0; v < H; ++v) {
        const float* row = depth_meters.ptr<float>(v);
        for (int u = 0; u < W; ++u) {
            float d = row[u];
            if (d > 0.0f && std::isfinite(d)) {
                valid_count++;
                sum_valid += d;
            } else if (d == 0.0f) {
                zero_count++;
            }
        }
    }

    double mean_valid = (valid_count > 0) ? (sum_valid / valid_count) : 0.0;

    // std::cout << "[DEPTH DEBUG] KF " << kf_id
    //           << " size=" << W << "x" << H
    //           << "  min=" << min_val
    //           << "  max=" << max_val
    //           << "  mean_valid=" << mean_valid
    //           << "  valid=" << valid_count
    //           << "  zeros=" << zero_count << "\n";
}

torch::Tensor VoxelMapper::computeSparseDepthLoss(
    const std::shared_ptr<VoxelKeyframe>& kf,
    const std::unordered_map<std::string, torch::Tensor>& render_pkg,
    int iteration)
{
    auto zero = torch::zeros(
        {1},
        torch::TensorOptions().dtype(torch::kFloat32).device(mDevice));

    // 0) Weight or schedule off -> no contribution
    if (opt_params_.lambda_sparse_depth_ <= 0.0f)
        return zero;

    loss_utils::SparseDepthLoss sparse_depth_loss(opt_params_.sparse_depth_until_);
    if (!sparse_depth_loss.isActive(iteration))
        return zero;

    // 1) Try to get raw_T / raw_depth (SVRaster-style) or fall back to T / depth (your wrapper)
    auto it_T = render_pkg.find("raw_T");
    auto it_depth = render_pkg.find("raw_depth");

    if (it_T == render_pkg.end() || it_depth == render_pkg.end() ||
        !it_T->second.defined() || !it_depth->second.defined())
    {
        std::cerr << "[SparseDepth] raw_T/raw_depth missing or undefined at iter "
                  << iteration
                  << " (did you forget to enable output_T/output_depth in RenderOpts?)\n";
        return zero;
    }

    torch::Tensor raw_T     = it_T->second.to(mDevice);
    torch::Tensor raw_depth = it_depth->second.to(mDevice);

    // 2) Build sparse_uv and sparse_depth from RGB-D for this keyframe
    torch::Tensor sparse_uv;     // [N,2]
    torch::Tensor sparse_depth;  // [N]
    if (!buildSparseDepthFromRGBD(kf, sparse_uv, sparse_depth)) {
        std::cout << "No valid depth points" << std::endl;
        // No valid sparse depth points for this keyframe
        // (no RGB-D or all invalid)
        return zero;
    }

    // 3) Low-level SparseDepthLoss (SVRaster math)
    torch::Tensor depth_loss =
        sparse_depth_loss(raw_T, raw_depth, sparse_uv, sparse_depth);

    // Optional debug
    // std::cout << "[SparseDepth] iter=" << iteration
    //           << " N=" << sparse_depth.size(0)
    //           << " loss=" << depth_loss.item<float>() << std::endl;

    return depth_loss;
}

// void VoxelMapper::applyFinalTsdfTransparency()
// {
//     if (!(sensor_type_ == RGBD && use_tsdf_mapping_)) {
//         std::cout << "[TSDF FINAL] RGBD/TSDF disabled, skipping final transparency.\n";
//         return;
//     }
//     if (!sdf_mapper_ || sdf_mapper_->tsdf_layer().size() == 0) {
//         std::cout << "[TSDF FINAL] no TSDF data, skipping final transparency.\n";
//         return;
//     }
//     std::unique_lock<std::mutex> lock_render(mutex_render_);

//     // ────────────────────────────────────────────────
//     // DEBUG: visualize *NVblox* TSDF voxels with distance < -0.15 (inside)
//     // ────────────────────────────────────────────────
//     {
//         nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
//         if (tsdf_layer.size() == 0) {
//             std::cout << "[TSDF DEBUG NVBLOX < 0.0] empty TSDF layer, skipping.\n";
//         } else {
//             const float block_size    = tsdf_layer.block_size();
//             const float voxel_size    = tsdf_layer.voxel_size();
//             const float inside_thresh = 0.0f;  // "definitely inside"
//             const float min_weight    = 1e-3f;   // consider as observed

//             using nvblox::Index3D;
//             using nvblox::TsdfBlock;
//             constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;

//             // 1) Enumerate all voxel centers in the TSDF layer
//             std::vector<Index3D> block_indices = tsdf_layer.getAllBlockIndices();
//             std::vector<nvblox::Vector3f> positions_L;
//             positions_L.reserve(
//                 block_indices.size() *
//                 kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide);

//             for (const Index3D& block_index : block_indices) {
//                 // NOTE: we do NOT dereference block->voxels here.
//                 for (int vx = 0; vx < kVoxelsPerSide; ++vx) {
//                     for (int vy = 0; vy < kVoxelsPerSide; ++vy) {
//                         for (int vz = 0; vz < kVoxelsPerSide; ++vz) {
//                             nvblox::Index3D voxel_index(vx, vy, vz);
//                             nvblox::Vector3f center =
//                                 nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
//                                     block_size, block_index, voxel_index);
//                             positions_L.push_back(center);
//                         }
//                     }
//                 }
//             }

//             if (positions_L.empty()) {
//                 std::cout << "[TSDF DEBUG NVBLOX < 0.0] no voxel centers, skipping.\n";
//             } else {
//                 // 2) Query TSDF values for all these centers (safe w.r.t. GPU)
//                 std::vector<nvblox::TsdfVoxel> voxels;
//                 std::vector<bool>              success_flags;
//                 tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

//                 // 3) Filter by TSDF value and weight, build visualization tensors
//                 std::vector<float> centers_vec;  // x,y,z
//                 std::vector<float> sizes_vec;    // edge length
//                 centers_vec.reserve(positions_L.size() * 3);
//                 sizes_vec.reserve(positions_L.size());

//                 const size_t M = voxels.size();
//                 for (size_t i = 0; i < M; ++i) {
//                     if (!success_flags[i]) continue;
//                     const auto& v = voxels[i];
//                     if (v.weight < min_weight) continue;
//                     if (v.distance > inside_thresh) continue;  // want < 0.0

//                     const auto& c = positions_L[i];
//                     centers_vec.push_back(c.x());
//                     centers_vec.push_back(c.y());
//                     centers_vec.push_back(c.z());
//                     sizes_vec.push_back(voxel_size);
//                 }

//                 const int64_t K = static_cast<int64_t>(centers_vec.size() / 3);
//                 std::cout << "[TSDF DEBUG NVBLOX < 0.0] K=" << K
//                           << " TSDF voxels with distance < "
//                           << inside_thresh << " (NVblox grid)\n";

//                 if (K > 0) {
//                     auto opts = torch::TensorOptions()
//                                     .dtype(torch::kFloat32)
//                                     .device(torch::kCPU);

//                     torch::Tensor centers_tsdf =
//                         torch::from_blob(centers_vec.data(), {K, 3}, opts).clone();
//                     torch::Tensor sizes_tsdf =
//                         torch::from_blob(sizes_vec.data(), {K, 1}, opts).clone();

//                     // [K,4] RGBA: red-ish for inside
//                     torch::Tensor colors_tsdf =
//                         torch::zeros({K, 4}, centers_tsdf.options());
//                     using torch::indexing::Slice;
//                     colors_tsdf.index_put_({Slice(), 0}, 1.0f);  // R
//                     colors_tsdf.index_put_({Slice(), 3}, 0.6f);  // alpha

//                     sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//                         centers_tsdf,
//                         sizes_tsdf,
//                         colors_tsdf,
//                         getIteration(),
//                         "world/nvblox_tsdf_voxels_neg015"
//                     );
//                 }
//             }
//         }
//     }
//     // ────────────────────────────────────────────────
//     // DEBUG: visualize *NVblox* TSDF voxels with distance == 0.0
//     // ────────────────────────────────────────────────
//     {
//         nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
//         if (tsdf_layer.size() == 0) {
//             std::cout << "[TSDF DEBUG NVBLOX 0.0] empty TSDF layer, skipping.\n";
//         } else {
//             const float block_size  = tsdf_layer.block_size();
//             const float voxel_size  = tsdf_layer.voxel_size();
//             const float target_tsdf = 0.0f;
//             const float min_weight  = 1e-3f;
//             const float eps         = 0.02f; 

//             using nvblox::Index3D;
//             using nvblox::TsdfBlock;
//             constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;

//             // 1) Build positions for ALL TSDF voxel centers
//             std::vector<Index3D>   block_indices = tsdf_layer.getAllBlockIndices();
//             std::vector<nvblox::Vector3f> positions_L;
//             positions_L.reserve(block_indices.size() * kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide);

//             for (const Index3D& block_index : block_indices) {
//                 TsdfBlock::ConstPtr block = tsdf_layer.getBlockAtIndex(block_index);
//                 if (!block) continue;

//                 for (int vx = 0; vx < kVoxelsPerSide; ++vx) {
//                     for (int vy = 0; vy < kVoxelsPerSide; ++vy) {
//                         for (int vz = 0; vz < kVoxelsPerSide; ++vz) {
//                             nvblox::Index3D voxel_index(vx, vy, vz);
//                             nvblox::Vector3f center =
//                                 nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
//                                     block_size, block_index, voxel_index);
//                             positions_L.push_back(center);
//                         }
//                     }
//                 }
//             }

//             if (positions_L.empty()) {
//                 std::cout << "[TSDF DEBUG NVBLOX 0.0] no voxel centers, skipping.\n";
//             } else {
//                 // 2) Query TSDF values for all these centers
//                 std::vector<nvblox::TsdfVoxel> voxels;
//                 std::vector<bool>              success_flags;
//                 tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

//                 // 3) Collect those with distance == 0.0 and enough weight
//                 std::vector<float> centers_vec;  // x,y,z
//                 std::vector<float> sizes_vec;    // one per voxel
//                 centers_vec.reserve(positions_L.size() * 3);
//                 sizes_vec.reserve(positions_L.size());

//                 const size_t M = voxels.size();
//                 for (size_t i = 0; i < M; ++i) {
//                     if (!success_flags[i]) continue;
//                     const auto& v = voxels[i];
//                     if (v.weight < min_weight) continue;
//                     // if (v.distance != target_tsdf) continue;
//                     // 2) Keep a *band* around zero, not exact equality
//                     if (std::fabs(v.distance - target_tsdf) > eps) continue;

//                     const auto& c = positions_L[i];
//                     centers_vec.push_back(c.x());
//                     centers_vec.push_back(c.y());
//                     centers_vec.push_back(c.z());
//                     sizes_vec.push_back(voxel_size);
//                 }

//                 const int64_t K = static_cast<int64_t>(centers_vec.size() / 3);
//                 std::cout << "[TSDF DEBUG NVBLOX 0] K=" << K
//                         << " TSDF voxels with distance == "
//                         << target_tsdf << " (NVblox grid)\n";

//                 if (K > 0) {
//                     auto opts = torch::TensorOptions()
//                                     .dtype(torch::kFloat32)
//                                     .device(torch::kCPU);

//                     torch::Tensor centers_tsdf =
//                         torch::from_blob(centers_vec.data(), {K, 3}, opts).clone();
//                     torch::Tensor sizes_tsdf =
//                         torch::from_blob(sizes_vec.data(), {K, 1}, opts).clone();

//                     torch::Tensor colors_tsdf =
//                         torch::zeros({K, 4}, centers_tsdf.options());
//                     using torch::indexing::Slice;
//                     colors_tsdf.index_put_({Slice(), 1}, 1.0f);  // G
//                     colors_tsdf.index_put_({Slice(), 2}, 1.0f);  // B
//                     colors_tsdf.index_put_({Slice(), 3}, 0.6f);  // alpha

//                     sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//                         centers_tsdf,
//                         sizes_tsdf,
//                         colors_tsdf,
//                         getIteration(),
//                         "world/nvblox_tsdf_voxels_0p0"
//                     );
//                 }
//             }
//         }
//     }
//     // ────────────────────────────────────────────────
//     // DEBUG: visualize *NVblox* TSDF voxels with distance > 0.15 (outside)
//     // ────────────────────────────────────────────────
//     {
//         nvblox::TsdfLayer& tsdf_layer = sdf_mapper_->tsdf_layer();
//         if (tsdf_layer.size() == 0) {
//             std::cout << "[TSDF DEBUG NVBLOX > 0.15] empty TSDF layer, skipping.\n";
//         } else {
//             const float block_size    = tsdf_layer.block_size();
//             const float voxel_size    = tsdf_layer.voxel_size();
//             const float inside_thresh = 0.15f;  // "definitely inside"
//             const float min_weight    = 1e-3f;   // consider as observed

//             using nvblox::Index3D;
//             using nvblox::TsdfBlock;
//             constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;

//             // 1) Enumerate all voxel centers in the TSDF layer
//             std::vector<Index3D> block_indices = tsdf_layer.getAllBlockIndices();
//             std::vector<nvblox::Vector3f> positions_L;
//             positions_L.reserve(
//                 block_indices.size() *
//                 kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide);

//             for (const Index3D& block_index : block_indices) {
//                 // NOTE: we do NOT dereference block->voxels here.
//                 for (int vx = 0; vx < kVoxelsPerSide; ++vx) {
//                     for (int vy = 0; vy < kVoxelsPerSide; ++vy) {
//                         for (int vz = 0; vz < kVoxelsPerSide; ++vz) {
//                             nvblox::Index3D voxel_index(vx, vy, vz);
//                             nvblox::Vector3f center =
//                                 nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(
//                                     block_size, block_index, voxel_index);
//                             positions_L.push_back(center);
//                         }
//                     }
//                 }
//             }

//             if (positions_L.empty()) {
//                 std::cout << "[TSDF DEBUG NVBLOX < 0.0] no voxel centers, skipping.\n";
//             } else {
//                 // 2) Query TSDF values for all these centers (safe w.r.t. GPU)
//                 std::vector<nvblox::TsdfVoxel> voxels;
//                 std::vector<bool>              success_flags;
//                 tsdf_layer.getVoxels(positions_L, &voxels, &success_flags);

//                 // 3) Filter by TSDF value and weight, build visualization tensors
//                 std::vector<float> centers_vec;  // x,y,z
//                 std::vector<float> sizes_vec;    // edge length
//                 centers_vec.reserve(positions_L.size() * 3);
//                 sizes_vec.reserve(positions_L.size());

//                 const size_t M = voxels.size();
//                 for (size_t i = 0; i < M; ++i) {
//                     if (!success_flags[i]) continue;
//                     const auto& v = voxels[i];
//                     if (v.weight < min_weight) continue;
//                     if (v.distance < inside_thresh) continue;  

//                     const auto& c = positions_L[i];
//                     centers_vec.push_back(c.x());
//                     centers_vec.push_back(c.y());
//                     centers_vec.push_back(c.z());
//                     sizes_vec.push_back(voxel_size);
//                 }

//                 const int64_t K = static_cast<int64_t>(centers_vec.size() / 3);
//                 std::cout << "[TSDF DEBUG NVBLOX < 0.0] K=" << K
//                           << " TSDF voxels with distance < "
//                           << inside_thresh << " (NVblox grid)\n";

//                 if (K > 0) {
//                     auto opts = torch::TensorOptions()
//                                     .dtype(torch::kFloat32)
//                                     .device(torch::kCPU);

//                     torch::Tensor centers_tsdf =
//                         torch::from_blob(centers_vec.data(), {K, 3}, opts).clone();
//                     torch::Tensor sizes_tsdf =
//                         torch::from_blob(sizes_vec.data(), {K, 1}, opts).clone();

//                     // [K,4] RGBA: red-ish for inside
//                     torch::Tensor colors_tsdf =
//                         torch::zeros({K, 4}, centers_tsdf.options());
//                     using torch::indexing::Slice;
//                     colors_tsdf.index_put_({Slice(), 0}, 1.0f);  // R
//                     colors_tsdf.index_put_({Slice(), 3}, 0.6f);  // alpha

//                     sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//                         centers_tsdf,
//                         sizes_tsdf,
//                         colors_tsdf,
//                         getIteration(),
//                         "world/nvblox_tsdf_voxels_p015"
//                     );
//                 }
//             }
//         }
//     }
//     // ────────────────────────────────────────────────
//     // sampleTsdfAtPointsWorld
//     // ────────────────────────────────────────────────
//     torch::Tensor centers_world = voxel_model_->voxCenter();  // [N,3]
//     if (!centers_world.defined() || centers_world.numel() == 0) {
//         std::cout << "[TSDF FINAL] no voxels, skipping final transparency.\n";
//         return;
//     }
//     const int N = centers_world.size(0);
//     // Sample TSDF for all voxel centers
//     torch::Tensor tsdf_vals = sampleTsdfAtPointsWorld(centers_world);  // [N]
//     // 1) finite mask
//     torch::Tensor tsdf_finite_mask = tsdf_vals.isfinite();  // [N]
//     if (!tsdf_finite_mask.any().item<bool>()) {
//         std::cout << "[TSDF FINAL] no finite TSDF values, skipping.\n";
//         return;
//     }
//     // // ────────────────────────────────────────────────
//     // // DEBUG: visualize voxels with TSDF == 0.2
//     // // ────────────────────────────────────────────────
//     // {
//     //     const float target_tsdf = 0.2f;    // exact value you want to inspect
//     //     // mask for finite TSDF AND exactly equal to target_tsdf
//     //     torch::Tensor eq_mask = tsdf_finite_mask & tsdf_vals.eq(target_tsdf); // [N]
//     //     eq_mask = eq_mask.to(torch::kBool);

//     //     auto n_eq = eq_mask.sum().item<int64_t>();
//     //     if (n_eq == 0) {
//     //         std::cout << "[TSDF DEBUG 0.2] no voxels with TSDF == "
//     //                 << target_tsdf << ", skipping visualization.\n";
//     //     } else {
//     //         // indices of those voxels
//     //         torch::Tensor idx_eq = eq_mask.nonzero().squeeze(1);  // [K]
//     //         const int64_t K = idx_eq.size(0);

//     //         // TSDF stats only on that exact set
//     //         torch::Tensor tsdf_eq_vals = tsdf_vals.index_select(0, idx_eq);  // [K]
//     //         float tsdf_eq_min  = tsdf_eq_vals.min().item<float>();
//     //         float tsdf_eq_max  = tsdf_eq_vals.max().item<float>();
//     //         float tsdf_eq_mean = tsdf_eq_vals.mean().item<float>();

//     //         std::cout << "[TSDF DEBUG 0.2] K=" << K
//     //                 << " tsdf_eq_min="  << tsdf_eq_min
//     //                 << " tsdf_eq_max="  << tsdf_eq_max
//     //                 << " tsdf_eq_mean=" << tsdf_eq_mean
//     //                 << " (TSDF == " << target_tsdf << ")\n";

//     //         // ---- build centers & sizes for visualization ----
//     //         torch::Tensor centers_0p2 =
//     //             centers_world.index_select(0, idx_eq).clone(); // [K,3]

//     //         torch::Tensor sizes_all = voxel_model_->voxSize();  // [N] or [N,1]
//     //         if (sizes_all.dim() == 1) {
//     //             sizes_all = sizes_all.view({N, 1});
//     //         } else if (sizes_all.dim() == 2 && sizes_all.size(1) == 1) {
//     //             // ok
//     //         } else {
//     //             sizes_all = sizes_all.reshape({N, 1});
//     //         }
//     //         torch::Tensor sizes_0p2 =
//     //             sizes_all.index_select(0, idx_eq).clone();       // [K,1]

//     //         // RGBA: bright green, semi-transparent
//     //         torch::Tensor colors_0p2 =
//     //             torch::zeros({K, 4}, centers_0p2.options());  // [K,4]
//     //         using torch::indexing::Slice;
//     //         colors_0p2.index_put_({Slice(), 1}, 1.0f);  // G channel
//     //         colors_0p2.index_put_({Slice(), 3}, 0.8f);  // alpha

//     //         std::cout << "[TSDF DEBUG 0.2] visualizing " << K
//     //                 << " voxels as 'world/voxels_tsdf_0p2' in rerun\n";

//     //         sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//     //             centers_0p2,
//     //             sizes_0p2,
//     //             colors_0p2,
//     //             getIteration(),
//     //             "world/voxels_tsdf_0p2"
//     //         );
//     //     }
//     // }

//     // 2) NVBlox freespace threshold
//     nvblox::FreespaceIntegrator freespace;
//     const float tsdf_free_thresh_m = freespace.max_tsdf_distance_for_occupancy_m(); //0.15
//     const float tsdf_margin_m      = 0.0f;
//     const float thresh             = tsdf_free_thresh_m + tsdf_margin_m;

//     torch::Tensor tsdf_free_mask = tsdf_vals > thresh;        // [N]
//     torch::Tensor tsdf_mask      = (tsdf_finite_mask & tsdf_free_mask).to(torch::kBool);

//     const int64_t n_tsdf = tsdf_mask.sum().item<int64_t>();
//     if (n_tsdf == 0) {
//         std::cout << "[TSDF FINAL] no voxels classified as free, nothing to fade.\n";
//         return;
//     }

//     // Optional stats (only over finite TSDF)
//     auto tsdf_finite = tsdf_vals.masked_select(tsdf_finite_mask);
//     float tsdf_min  = tsdf_finite.min().item<float>();
//     float tsdf_max  = tsdf_finite.max().item<float>();
//     float tsdf_mean = tsdf_finite.mean().item<float>();
//     std::cout << "[TSDF FINAL] N=" << N
//               << " tsdf_min="  << tsdf_min
//               << " tsdf_max="  << tsdf_max
//               << " tsdf_mean=" << tsdf_mean
//               << " free_mask_sum=" << n_tsdf
//               << " thresh=" << thresh
//               << "\n";
//     // ------------------------------------------------------------------
//     // DEBUG MODE: operate on ONE voxel: the furthest free-space voxel
//     // ------------------------------------------------------------------
//     {
//         // indices of free-space voxels
//         torch::Tensor idx_free = tsdf_mask.nonzero().squeeze(1);  // [K]
//         TORCH_CHECK(idx_free.numel() > 0, "tsdf_mask has no true entries unexpectedly.");

//         // TSDF values only for free voxels
//         torch::Tensor tsdf_free_vals = tsdf_vals.index_select(0, idx_free); // [K]

//         // argmax over that subset
//         auto max_pair      = tsdf_free_vals.max(0);                  // (values, indices)
//         int64_t local_arg  = std::get<1>(max_pair).item<int64_t>();  // index in [0..K-1]
//         int64_t voxel_idx  = idx_free[local_arg].item<int64_t>();    // index in [0..N-1]

//         float tsdf_vox = tsdf_vals[voxel_idx].item<float>();
//         std::cout << "[TSDF DEBUG] picked voxel_idx=" << voxel_idx
//                   << " with tsdf=" << tsdf_vox << " (max over free-space set)\n";

//         // --- visualize this voxel in Rerun ---
//         // Make a 1-element index tensor
//         auto idx_options = torch::TensorOptions().dtype(torch::kLong).device(centers_world.device());
//         torch::Tensor idx_single = torch::tensor({voxel_idx}, idx_options); // [1]

//         torch::Tensor center_debug = centers_world.index_select(0, idx_single); // [1,3]

//         torch::Tensor size_all = voxel_model_->voxSize(); // [N] or [N,1]
//         if (size_all.dim() == 1) {
//             size_all = size_all.view({N, 1});
//         } else if (size_all.dim() == 2 && size_all.size(1) == 1) {
//             // ok
//         } else {
//             size_all = size_all.reshape({N, 1});
//         }
//         torch::Tensor size_debug = size_all.index_select(0, idx_single); // [1,1]

//         // RGBA: red, semi-transparent
//         torch::Tensor colors_debug = torch::zeros({1, 4}, center_debug.options());
//         colors_debug.index_put_({0, 0}, 1.0f);  // R
//         colors_debug.index_put_({0, 3}, 0.8f);  // alpha

//         sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//             center_debug,
//             size_debug,
//             colors_debug,
//             getIteration(),
//             "world/tsdf_debug_voxel"  // separate entity
//         );

//         // ---- make ONLY this voxel transparent via its 8 grid points ----
//         const float geo_value_tsdf_free = -30.0f;  // or whatever you're using
//         voxel_model_->applySingleVoxelTsdfTransparency(voxel_idx, geo_value_tsdf_free);

//         std::cout << "[TSDF DEBUG] applied transparency to single voxel_idx="
//                   << voxel_idx << "\n";

//         // IMPORTANT: early return here so we don't call the bulk path
//         // return;
//     }

//     // Push log-density of those free voxels down to a very low value
//     const float geo_value_tsdf_free = -30.0f;   //
//     voxel_model_->applyTsdfTransparency(tsdf_mask, geo_value_tsdf_free);

//     // --- OPTIONAL: debug visualization of TSDF-free voxels as a separate entity ---
//     try {
//         auto idx = tsdf_mask.nonzero().squeeze(1);   // [K]
//         if (idx.numel() > 0) {
//             torch::Tensor centers_tsdf = centers_world.index({idx}).clone();  // [K,3]
//             torch::Tensor sizes_tsdf   = voxel_model_->voxSize();             // [N] or [N,1]
//             if (sizes_tsdf.dim() == 1) {
//                 sizes_tsdf = sizes_tsdf.view({N, 1});
//             } else if (sizes_tsdf.dim() == 2 && sizes_tsdf.size(1) == 1) {
//                 // ok
//             } else {
//                 sizes_tsdf = sizes_tsdf.reshape({N, 1});
//             }
//             sizes_tsdf = sizes_tsdf.index({idx}).clone();                     // [K,1]

//             const auto K = centers_tsdf.size(0);
//             torch::Tensor colors_tsdf = torch::zeros({K, 4}, centers_tsdf.options());
//             // e.g. blue-transparent
//             colors_tsdf.index_put_({torch::indexing::Slice(), 2}, 1.0f);  // B
//             colors_tsdf.index_put_({torch::indexing::Slice(), 3}, 0.7f);  // alpha

//             std::cout << "[TSDF FINAL] visualizing " << K
//                       << " TSDF-free voxels as 'world/voxels_tsdf' in rerun\n";

//             sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
//                 centers_tsdf,
//                 sizes_tsdf,
//                 colors_tsdf,
//                 getIteration(),           // current iter
//                 "world/voxels_tsdf_transparent"       // separate entity
//             );
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "[TSDF FINAL] rerun visualization error: " << e.what() << "\n";
//     }
// }

// Add training with optimization loop
void VoxelMapper::trainForOneIteration()
{
    // 1) bump global iteration counter
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    sv::RenderOpts ropts;

    // 2) pick a random keyframe from the sliding window
    std::shared_ptr<VoxelKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }
    writeKeyframeUsedTimes(result_dir_ / "used_times");

    const int iter = getIteration();
    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;

    if (isdoingGausPyramidTraining())
         training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        // std::cout << "training full res\n";
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_
                            .to(mDevice);          // (3,H,W)
        mask = undistort_mask_[viewpoint_cam->camera_id_]
                                    .to(mDevice)
                                    .to(torch::kFloat32); // (3,H,W)
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].to(mDevice); 
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level].to(mDevice).to(torch::kFloat32);
    }
    // std::cout << "gt_image" << gt_image << std::endl;
    // 4) grow SH degree every 1000 iterations (locked during render)
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
    {
        default_sh_ += 1;
        std::cout << "[VoxelMapper] SH degree: " << default_sh_ << std::endl;
    }    
    voxel_model_->setShDegree(default_sh_);

    // Use default super-sampling option (enable after 1000 iters)
    // if (iter > 200) {
    //     if (opt_params_.ss_aug_max_ > 1.0f) {
    //         static thread_local std::mt19937 rng{std::random_device{}()};
    //         std::uniform_real_distribution<float> dist(1.0f, opt_params_.ss_aug_max_);
    //         ropts.ss = dist(rng);                 // tr_render_opt['ss'] = U(1, ss_aug_max)
    //     } else {
    //         ropts.ss = std::nullopt;              // pop('ss') -> use model default self.ss
    //     }
    // } else {
    //     ropts.ss = 1.0f;                           // disable supersampling at first
    // }
    // ropts.ss = 1.0f;   

    const bool need_sparse_depth = (opt_params_.lambda_sparse_depth_ > 0.0f) && (iter <= opt_params_.sparse_depth_until_);
    ropts.output_T     = (opt_params_.lambda_T_inside_ > 0.0f) || need_sparse_depth;
    ropts.output_depth = need_sparse_depth; 

    // if (opt_params_.lambda_T_inside_ > 0.0f) {
    //     ropts.output_T = true;
    // }

    if (iter > 400 && opt_params_.lambda_dist_ > 0.0f) {
        ropts.lambda_dist = opt_params_.lambda_dist_;
    }

    if (opt_params_.lambda_R_concen_ > 0.0f) {
        ropts.lambda_R_concen = opt_params_.lambda_R_concen_;
        ropts.gt_color = gt_image;
    }

    // ) build a MiniCam out of this keyframe
    // std::cout << "build minicam image size: " << image_width << "x" << image_height << std::endl;
    sv::MiniCam cam = viewpoint_cam->toMiniCam(image_height, image_width);
    // tr_cams.push_back(cam); // for densification later

    // std::cout << "ropts.track_max_w = " << ropts.track_max_w << std::endl;
    auto render_pkg = voxel_model_->render(
        cam,
        image_height,
        image_width,
        /* gt_image   */  gt_image,            
        /* color_mode   */   nullptr,             
        /* track_max_w   */  true,
        /* ss            */  std::nullopt,
        /* output_depth  */  true,
        /* output_normal */  false,
        /* output_T      */  ropts.output_T,
        /* rand_bg       */  false,
        /* use_auto_exp  */  false,
        ropts               // your struct (will be used for **other_opt-safe fields)
    );
    if (render_pkg.empty() || !render_pkg.count("color") || !render_pkg.at("color").defined()) {
        std::cout << "voxel_mapper render: pkg empty" << std::endl;
        return;
    }
    // // keep running max_w stats for pruning diagnostics (if returned)
    // if (render_pkg.count("max_w") && render_pkg.at("max_w").defined()) {
    //     voxel_model_->max_w_ = torch::maximum(
    //         voxel_model_->max_w_, render_pkg["max_w"].to(mDevice));
    // }

    torch::Tensor rendered_image = render_pkg["color"].to(mDevice);
    torch::Tensor masked_image = rendered_image * mask;      // (1,3,H,W)
    // torch::Tensor masked_gt   = gt_image * mask;
    if (!rendered_image.requires_grad()) {
        std::cerr << "[warn] rendered_image.requires_grad == false; grad_fn="
                << (rendered_image.grad_fn() ? "set" : "NULL") << "\n";
    }

    // after render_pkg & rendered_image
    torch::Tensor depth_for_viz;   // declare here so it's visible later
    auto it_depth = render_pkg.find("depth");
    if (it_depth != render_pkg.end() && it_depth->second.defined()) {
        depth_for_viz = it_depth->second;  // keep on device for now
    }

    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto photoslam_loss = (1.0 - lambda_dssim) * Ll1
            + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, mDevice.type()));
    auto mse  = loss_utils::l2_loss(masked_image, gt_image);
    auto loss = mse.clone(); 

    // --- Sparse depth regularization (SVRaster-style) ----------------------------
    if (need_sparse_depth) {
        torch::Tensor depth_loss =
            // computeSparseDepthLoss(viewpoint_cam, render_pkg, iter);
            computeSparseDepthLoss_Points(
            viewpoint_cam,   // which KF we are training on
            cam,             // MiniCam for this KF at current pyramid level
            image_width,
            image_height,
            render_pkg,
            iter);

        float dl = depth_loss.item<float>();
        // if (dl > 0.0f || iter % 100 == 0) {
        //     std::cout << "[iter " << iter << "] sparse depth loss = " << dl << std::endl;
        // }
        loss = loss + opt_params_.lambda_sparse_depth_ * depth_loss;
    }
    if (opt_params_.lambda_ssim_ > 0.0f) {
        loss += opt_params_.lambda_ssim_ * loss_utils::fast_ssim_loss(masked_image, gt_image);
        // std::cout << "[iter " << iter << "] "
        //           << "MSE: " << mse.item<float>()
        //           << " SSIM_loss: " << (opt_params_.lambda_ssim_ * loss_utils::fast_ssim_loss(masked_image, gt_image)).item<float>()
        //           << " loss: " << loss.item<float>() << "\n";
    }

    if (opt_params_.lambda_T_inside_ > 0.0f) {
        auto it = render_pkg.find("raw_T");
        if (it != render_pkg.end() && it->second.defined()) {
            // raw_T has shape (…, H, W); same as in Python
            torch::Tensor reg = it->second.pow(2).mean();
            loss = loss + opt_params_.lambda_T_inside_ * reg;
        } else {
            // We expected raw_T because we requested output_T above
            std::cerr << "[warn] raw_T not in render_pkg (output_T might be off)\n";
        }
    }

    voxel_model_->optimizerZeroGrad();   // move this BEFORE backward
    {
        py::gil_scoped_release no_gil;
        loss.backward();
    }

    if (opt_params_.lambda_tv_density_ > 0.f &&
        iter >= opt_params_.tv_from_ &&
        iter <= opt_params_.tv_until_) {
        voxel_model_->applyTvOnDensityField(opt_params_.lambda_tv_density_);
    }

    voxel_model_->optimizerStep();   // <-- the actual update

    // --- debug: store near voxels for rerun ---
    torch::Tensor debug_near_centers;  // [K,3]
    torch::Tensor debug_near_sizes;    // [K,1] or [K]
    bool debug_has_near = false;
    torch::Tensor debug_tsdf_centers;   // [K_tsdf,3]
    torch::Tensor debug_tsdf_sizes;     // [K_tsdf,1] or [K_tsdf]
    bool debug_has_tsdf = false;
    {
        // Densification for increasePcd
        const bool meet_adapt_period =
            (iter % opt_params_.adapt_every_ == 0) &&
            (iter >= opt_params_.adapt_from_);
            // (iter == opt_params_.adapt_from_);

        bool need_pruning =
            meet_adapt_period && (iter <= opt_params_.prune_until_);

        bool need_subdividing =
            meet_adapt_period &&
            (iter <= opt_params_.subdivide_until_) &&
            (voxel_model_->numVoxels() < opt_params_.subdivide_max_num_);

        if (need_pruning || need_subdividing)
        {
            // // NEW: cooldown after fill_empty_cells_ artifact creation
            // constexpr int64_t kMinItersAfterFill = 200;  // tune or move to YAML
            // if (last_artifact_fill_iter_ >= 0) {
            //     int64_t dt = static_cast<int64_t>(iter) - last_artifact_fill_iter_;
            //     if (dt < kMinItersAfterFill) {
            //         std::cout << "[VoxelMapper] skipping prune/subdiv at iter "
            //                 << iter << " (dt=" << dt
            //                 << " < " << kMinItersAfterFill
            //                 << " since last artifact fill)\n";
            //         // Skip densification for this iteration, but keep training, etc.
            //         return;   // exit trainForOneIteration() here
            //     }
            // }
            // // Build list of training cameras (use all current keyframes)
            std::vector<sv::MiniCam> tr_cams; 
            tr_cams.reserve(scene_->keyframes().size());
            std::cout << "keyframes size: " << scene_->keyframes().size() << std::endl;
            // std::cout << "image size: " << image_width << "x" << image_height << std::endl;
            for (auto& kv : scene_->keyframes()) {
                // std::cout << "keyframe id: " << kv.first << std::endl;
                // std::cout << "image_height_ : " << kv.second->image_height_ << std::endl;
                if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
            }
            
            // Compute statistics once (max_w, min_samp_interval, view_cnt)
            // std::cout << "length of training cams: " << tr_cams.size() << std::endl;

            // tr_cams.reserve(1);
            // tr_cams.push_back(cam);
            auto stat = voxel_model_->computeTrainingStat(tr_cams);
            py::object sched_state = voxel_model_->schedulerStateDict();

            std::cout << "[densify:stat] N=" << voxel_model_->numVoxels()
                    << "  max_w=" << shp(stat.max_w)
                    << "  min_itv=" << shp(stat.min_samp_interval)
                    << "  view_cnt=" << shp(stat.view_cnt) << "\n";

            // ---------------- PRUNE ----------------
            if (need_pruning) {
                const int   a0 = opt_params_.adapt_from_;
                const int   a1 = opt_params_.prune_until_;
                const float t0 = opt_params_.prune_thres_init_;
                const float t1 = opt_params_.prune_thres_final_;

                const float prune_thres = (iter <= a0) ? t0 :
                                        (iter >= a1) ? t1 :
                                        (t0 + (t1 - t0) * float(iter - a0) / float(std::max(1, a1 - a0)));
                // float prune_thres = 0.15f; // fixed threshold
                const int ori_n = voxel_model_->numVoxels();
                auto prune_mask = (stat.max_w < prune_thres).squeeze(1); // [N] bool
                const int N     = ori_n;

                // --- 1) base Photo-SLAM / SVRaster pruning (max_w) ---
                torch::Tensor prune_mask_base =
                    (stat.max_w < prune_thres).squeeze(1);      // [N] bool or byte
                prune_mask_base = prune_mask_base.to(torch::kBool);
                const int n_prune_base =
                    prune_mask_base.defined()
                        ? (int)prune_mask_base.sum().item<int64_t>()
                        : -1;
                std::cout << "[PRUNE:base] thresh=" << prune_thres
                        << "  N=" << N
                        << " base_prune_sum=" << n_prune_base << "\n";

                int n_prune_tsdf   = 0;
                int n_prune_union  = n_prune_base;
                int n_prune_tsdf_only = 0;
                int n_prune_overlap   = 0;
                // NEW: declare tsdf_prune_mask here, default undefined
                torch::Tensor tsdf_prune_mask;
                // ---------------- OPTIONAL: TSDF-guided pruning ----------------
                // if (sensor_type_ == RGBD && use_tsdf_mapping_) {
                //     const bool tsdf_prune_active = (iter >= 500);   // gate (move to YAML later)
                //     if (tsdf_prune_active && N > 0 && sdf_mapper_ && sdf_mapper_->tsdf_layer().size() > 0) {
                //         try {
                //             torch::Tensor centers_world = voxel_model_->voxCenter();  // [N,3]

                //             if (centers_world.defined() &&
                //                 centers_world.dim() == 2 &&
                //                 centers_world.size(0) == N &&
                //                 centers_world.size(1) == 3)
                //             {
                //                 TsdfSample s = sampleTsdfAtPointsWorld(centers_world);
                //                 torch::Tensor tsdf_vals   = s.tsdf;     // [N] float
                //                 torch::Tensor tsdf_weight = s.weight;   // [N] float
                //                 torch::Tensor tsdf_has    = s.success;  // [N] bool

                //                 // Make sure everything lives on the same device as prune_mask
                //                 if (tsdf_vals.device() != prune_mask.device()) {
                //                     tsdf_vals   = tsdf_vals.to(prune_mask.device());
                //                     tsdf_weight = tsdf_weight.to(prune_mask.device());
                //                     tsdf_has    = tsdf_has.to(prune_mask.device());
                //                 }

                //                 // (1) Validity gate: only trust TSDF where lookup succeeded AND weight is meaningful
                //                 // Tune min_weight; 1e-3 is very permissive. Often 1e-2 ... 1e-1 is more stable.
                //                 const float min_tsdf_weight = 1e-3f;  // TODO: move to YAML
                //                 torch::Tensor tsdf_valid_mask =
                //                     tsdf_has & (tsdf_weight >= min_tsdf_weight) & tsdf_vals.isfinite();   // [N] bool

                //                 // (2) Threshold for free space (NVBlox freespace logic).
                //                 nvblox::FreespaceIntegrator freespace;
                //                 const float tsdf_free_thresh_m =
                //                     freespace.max_tsdf_distance_for_occupancy_m();

                //                 const float tsdf_margin_m = 0.0f;
                //                 const float prune_tsdf_thresh = tsdf_free_thresh_m + tsdf_margin_m;

                //                 // (3) Free-space mask: TSDF clearly in front of surface
                //                 torch::Tensor tsdf_free_mask = tsdf_vals > prune_tsdf_thresh;  // [N] bool

                //                 // (4) Final TSDF prune mask: only prune where TSDF is valid and indicates free space
                //                 tsdf_prune_mask = (tsdf_valid_mask & tsdf_free_mask).to(torch::kBool);  // [N] bool

                //                 // Stats on TSDF values (only where valid)
                //                 if (tsdf_valid_mask.any().item<bool>()) {
                //                     auto tsdf_valid = tsdf_vals.masked_select(tsdf_valid_mask);
                //                     auto w_valid    = tsdf_weight.masked_select(tsdf_valid_mask);

                //                     const float tsdf_min  = tsdf_valid.min().item<float>();
                //                     const float tsdf_max  = tsdf_valid.max().item<float>();
                //                     const float tsdf_mean = tsdf_valid.mean().item<float>();

                //                     const float w_min  = w_valid.min().item<float>();
                //                     const float w_max  = w_valid.max().item<float>();
                //                     const float w_mean = w_valid.mean().item<float>();

                //                     n_prune_tsdf = (int)tsdf_prune_mask.sum().item<int64_t>();

                //                     std::cout << "[TSDF PRUNE] iter=" << iter
                //                             << " valid=" << tsdf_valid.numel() << "/" << N
                //                             << " tsdf(min/mean/max)=" << tsdf_min << "/" << tsdf_mean << "/" << tsdf_max
                //                             << " w(min/mean/max)=" << w_min << "/" << w_mean << "/" << w_max
                //                             << " prune_tsdf_sum=" << n_prune_tsdf
                //                             << " thresh=" << prune_tsdf_thresh
                //                             << " (free_th=" << tsdf_free_thresh_m << ", margin=" << tsdf_margin_m << ")"
                //                             << std::endl;
                //                 } else {
                //                     std::cout << "[TSDF PRUNE] iter=" << iter
                //                             << " no valid TSDF (success+weight) values, skipping TSDF-based pruning.\n";
                //                     // ensure tsdf_prune_mask is at least defined and correct shape
                //                     tsdf_prune_mask = torch::zeros({N}, prune_mask.options().dtype(torch::kBool));
                //                 }

                //                 // (5) Union + overlap statistics
                //                 auto prune_mask_union = prune_mask | tsdf_prune_mask;      // [N]
                //                 auto overlap_mask     = prune_mask_base & tsdf_prune_mask; // [N]

                //                 n_prune_union     = (int)prune_mask_union.sum().item<int64_t>();
                //                 n_prune_overlap   = (int)overlap_mask.sum().item<int64_t>();
                //                 n_prune_tsdf_only = n_prune_union - n_prune_base;

                //                 std::cout << "[PRUNE:tsdf_stats] N=" << N
                //                         << " base=" << n_prune_base
                //                         << " tsdf=" << n_prune_tsdf
                //                         << " union=" << n_prune_union
                //                         << " tsdf_only=" << n_prune_tsdf_only
                //                         << " overlap=" << n_prune_overlap
                //                         << "\n";

                //                 // Debug visualization (only TSDF-pruned)
                //                 auto tsdf_idx = tsdf_prune_mask.nonzero().squeeze(1);  // [K_tsdf]
                //                 if (tsdf_idx.numel() > 0) {
                //                     debug_tsdf_centers = centers_world.index({tsdf_idx}).clone();  // [K_tsdf,3]

                //                     torch::Tensor all_vox_size = voxel_model_->voxSize();          // [N] or [N,1]
                //                     if (all_vox_size.device() != centers_world.device()) {
                //                         all_vox_size = all_vox_size.to(centers_world.device());
                //                     }
                //                     if (all_vox_size.dim() == 1) {
                //                         all_vox_size = all_vox_size.view({N, 1});
                //                     } else if (!(all_vox_size.dim() == 2 && all_vox_size.size(1) == 1)) {
                //                         all_vox_size = all_vox_size.reshape({N, 1});
                //                     }
                //                     debug_tsdf_sizes = all_vox_size.index({tsdf_idx}).clone();     // [K_tsdf,1]
                //                     debug_has_tsdf = true;

                //                     std::cout << "[DEBUG TSDF] saved " << tsdf_idx.size(0)
                //                             << " TSDF-pruned voxels for rerun visualization\n";
                //                 } else {
                //                     debug_has_tsdf = false;
                //                 }

                //                 // const float geo_value_tsdf_free = -10.0f;   // or -12.0f if you want them super transparent
                //                 // if (tsdf_prune_mask.defined() && tsdf_prune_mask.any().item<bool>()) {
                //                 //     voxel_model_->applyTsdfTransparency(tsdf_prune_mask, geo_value_tsdf_free);
                //                 //     std::cout << "[TSDF TRANSPARENCY] applied to "
                //                 //             << tsdf_prune_mask.sum().item<int64_t>()
                //                 //             << " voxels (set geo to " << geo_value_tsdf_free << ")\n";
                //                 // } //add this in the end of training

                //                 // 6) Use the union as final prune mask
                //                 prune_mask = prune_mask_union;

                //                 // // Everything with TSDF > 0 is free or unknown -> prune.    
                //                 // tsdf_prune_mask = (tsdf_vals > 0.0f).to(torch::kBool);
                //                 // int n_prune_tsdf = (int)tsdf_prune_mask.sum().item<int64_t>();
                //                 // std::cout << "[TSDF PRUNE] iter=" << iter
                //                 //         << " tsdf_vals: min=" << tsdf_vals.min().item<float>()
                //                 //         << " max="          << tsdf_vals.max().item<float>()
                //                 //         << " mean="         << tsdf_vals.mean().item<float>()
                //                 //         << " prune_tsdf_sum=" << n_prune_tsdf
                //                 //         << "\n";
                //                 // auto prune_mask_union = prune_mask | tsdf_prune_mask;
                //                 // prune_mask = prune_mask_union;
                //                 // // float mu = tsdf_vals.abs().max().item<float>();
                //                 // // if (mu > 0.0f && !std::isnan(mu)) {
                //                 // //     const float tau_surface = 0.1f * mu;
                //                 // //     const float tau_free    = 0.9f * mu;
                //                 // //     auto tsdf_abs          = tsdf_vals.abs();
                //                 // //     auto near_surface_mask = (tsdf_abs < tau_surface);   // |φ| < tau_surface
                //                 // //     auto tsdf_free_mask    = (tsdf_vals > tau_free);     // clearly front of surface
                //                 // //     tsdf_prune_mask = tsdf_free_mask & (~near_surface_mask); // [N] bool
                //                 // //     tsdf_prune_mask = tsdf_prune_mask.to(prune_mask.device());
                //                 // //     n_prune_tsdf = (int)tsdf_prune_mask.sum().item<int64_t>();
                //                 // //     std::cout << "[TSDF PRUNE] iter=" << iter
                //                 // //             << " TSDF stats: min=" << tsdf_vals.min().item<float>()
                //                 // //             << " max=" << tsdf_vals.max().item<float>()
                //                 // //             << " mean=" << tsdf_vals.mean().item<float>()
                //                 // //             << " mu=" << mu
                //                 // //             << " tau_surface=" << tau_surface
                //                 // //             << " tau_free=" << tau_free
                //                 // //             << " tsdf_prune_sum=" << n_prune_tsdf
                //                 // //             << "\n";
                //                 // //     // union and overlap statistics
                //                 // //     auto prune_mask_union = prune_mask | tsdf_prune_mask;          // [N]
                //                 // //     auto overlap_mask     = prune_mask_base & tsdf_prune_mask;     // [N]
                //                 // //     n_prune_union     = (int)prune_mask_union.sum().item<int64_t>();
                //                 // //     n_prune_overlap   = (int)overlap_mask.sum().item<int64_t>();
                //                 // //     n_prune_tsdf_only = n_prune_union - n_prune_base;
                //                 // //     std::cout << "[PRUNE:tsdf_stats] N=" << N
                //                 // //             << " base=" << n_prune_base
                //                 // //             << " tsdf=" << n_prune_tsdf
                //                 // //             << " union=" << n_prune_union
                //                 // //             << " tsdf_only=" << n_prune_tsdf_only
                //                 // //             << " overlap=" << n_prune_overlap
                //                 // //             << "\n";
                //                 // //     // use the union as final prune mask
                //                 // //     prune_mask = prune_mask_union;
                //                 // // } else {
                //                 // //     std::cout << "[TSDF PRUNE] mu invalid (" << mu
                //                 // //             << "), skipping TSDF-based pruning.\n";
                //                 // // }
                //             } else {
                //                 std::cout << "[TSDF PRUNE] centers_world shape mismatch, skipping TSDF.\n";
                //             }
                //         } catch (const std::exception& e) {
                //             std::cerr << "[TSDF PRUNE] exception: " << e.what() << "\n";
                //         }
                //     }
                // }
                
                if (sensor_type_ == RGBD && use_tsdf_mapping_) {
                    const bool tsdf_prune_active = (iter >= 500);
                    if (tsdf_prune_active && N > 0 && sdf_mapper_ && sdf_mapper_->tsdf_layer().size() > 0) {
                        try {
                            torch::Tensor centers_world = voxel_model_->voxCenter(); // [N,3]
                            torch::Tensor sizes_world   = voxel_model_->voxSize();   // [N,1] (your implementation)

                            if (centers_world.defined() &&
                                centers_world.dim() == 2 &&
                                centers_world.size(0) == N &&
                                centers_world.size(1) == 3 &&
                                sizes_world.defined() &&
                                sizes_world.size(0) == N)
                            {
                                // Sample TSDF at 8 corners per voxel
                                // TsdfCornerSample c = sampleTsdfAtVoxelCornersWorld(centers_world, sizes_world);
                                TsdfCornerSample c = sampleTsdfAtSvrasterGridCornersWorld();
                                torch::Tensor tsdf8   = c.tsdf;    // [N,8]
                                torch::Tensor w8      = c.weight;  // [N,8]
                                torch::Tensor ok8     = c.success; // [N,8] bool

                                // Device alignment (should already match)
                                if (tsdf8.device() != prune_mask.device()) {
                                    tsdf8 = tsdf8.to(prune_mask.device());
                                    w8    = w8.to(prune_mask.device());
                                    ok8   = ok8.to(prune_mask.device());
                                }

                                // Weight threshold: only trust observed TSDF
                                const float min_weight = 1e-3f;  // move to YAML later if desired
                                torch::Tensor w_ok8 = (w8 >= min_weight);
                                // Corner valid if both success and sufficient weight
                                torch::Tensor corner_valid = ok8 & w_ok8;   // [N,8] bool
                                // Strict voxel validity: require all 8 corners valid
                                torch::Tensor voxel_valid = corner_valid.all(/*dim=*/1); // [N] bool

                                // Sign test with epsilon: values in [-eps, eps] count as "near surface" -> prevent pruning
                                const float eps = 1e-4f;  // meters, small; can scale with voxel size if you prefer
                                torch::Tensor all_pos = (tsdf8 >  eps).all(/*dim=*/1);  // [N]
                                torch::Tensor all_neg = (tsdf8 < -eps).all(/*dim=*/1);  // [N]
                                torch::Tensor same_sign = all_pos | all_neg;            // [N] no zero-crossing inside cell

                                // ------------------ Far-from-surface gating (NEW) ------------------
                                // Require that all corners are sufficiently far from 0 (confidently empty/inside).
                                // A robust default is tied to NVBlox TSDF voxel resolution.
                                const float k_far = 1.0f;  // 1x tsdf voxel size (tune: 0.5..2.0)
                                const float tau_far = k_far * sdf_mapper_->tsdf_layer().voxel_size(); // meters
                                // torch::Tensor far_enough = (tsdf8.abs() > tau_far).all(/*dim=*/1);               // [N]
                                nvblox::FreespaceIntegrator freespace;
                                const float tsdf_free_thresh_m = freespace.max_tsdf_distance_for_occupancy_m();
                                torch::Tensor far_enough = (tsdf8.abs() > tsdf_free_thresh_m).all(/*dim=*/1);               // [N]

                                // TSDF prune mask: valid corners AND no zero-crossing
                                tsdf_prune_mask = (voxel_valid & same_sign & far_enough).to(torch::kBool); // [N]

                                // Ensure device matches prune_mask
                                if (tsdf_prune_mask.device() != prune_mask.device()) {
                                    tsdf_prune_mask = tsdf_prune_mask.to(prune_mask.device());
                                }

                                // Stats
                                n_prune_tsdf = (int)tsdf_prune_mask.sum().item<int64_t>();

                                // Additional diagnostics (optional, but useful early)
                                {
                                    int64_t n_valid = voxel_valid.sum().item<int64_t>();
                                    int64_t n_same  = same_sign.sum().item<int64_t>();
                                    std::cout << "[TSDF CORNER PRUNE] iter=" << iter
                                            << " N=" << N
                                            << " voxel_valid=" << n_valid
                                            << " same_sign=" << n_same
                                            << " prune_tsdf_sum=" << n_prune_tsdf
                                            << " min_weight=" << min_weight
                                            << " eps=" << eps
                                            << " tau_far=" << tau_far
                                            << std::endl;
                                }

                                // Union + overlap statistics
                                auto prune_mask_union = prune_mask | tsdf_prune_mask;      // [N]
                                auto overlap_mask     = prune_mask_base & tsdf_prune_mask; // [N]

                                n_prune_union     = (int)prune_mask_union.sum().item<int64_t>();
                                n_prune_overlap   = (int)overlap_mask.sum().item<int64_t>();
                                n_prune_tsdf_only = n_prune_union - n_prune_base;

                                std::cout << "[PRUNE:tsdf_stats] N=" << N
                                        << " base=" << n_prune_base
                                        << " tsdf=" << n_prune_tsdf
                                        << " union=" << n_prune_union
                                        << " tsdf_only=" << n_prune_tsdf_only
                                        << " overlap=" << n_prune_overlap
                                        << "\n";

                                // Save debug voxels pruned by TSDF for rerun visualization
                                auto tsdf_idx = tsdf_prune_mask.nonzero().squeeze(1); // [K]
                                if (tsdf_idx.numel() > 0) {
                                    debug_tsdf_centers = centers_world.index({tsdf_idx}).clone(); // [K,3]
                                    // sizes_world is [N,1] already
                                    debug_tsdf_sizes   = sizes_world.index({tsdf_idx}).clone();   // [K,1]
                                    debug_has_tsdf     = true;
                                    std::cout << "[DEBUG TSDF] saved " << tsdf_idx.size(0)
                                            << " TSDF-pruned voxels for rerun visualization\n";
                                } else {
                                    debug_has_tsdf = false;
                                }

                                // Use union as final prune mask
                                prune_mask = prune_mask_union;
                            } else {
                                std::cout << "[TSDF CORNER PRUNE] centers/sizes shape mismatch, skipping.\n";
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "[TSDF CORNER PRUNE] exception: " << e.what() << "\n";
                        }
                    }
                }

                // ---------------- NEW: SVRaster-like VISIBILITY / NEAR filtering ----------------
                // This mimics octlayout_filtering(...) using mark_max_samp_rate + mark_near,
                // but we apply it via pruning on the current model layout.
                if (!tr_cams.empty() && N > 0) {
                    try {
                        py::gil_scoped_acquire gil;
                        static py::module_ svr_mod =
                            py::module_::import("svraster_cuda").attr("renderer");
                        static py::module_ torch_mod =
                            py::module_::import("torch");

                        // Access Python SparseVoxelModel
                        py::object py_svm = voxel_model_->svm();
                        if (!py_svm.is_none()) {

                            py::object py_octpath   = py_svm.attr("octpath");
                            py::object py_octlv     = py_svm.attr("octlevel");
                            py::object py_vox_center= py_svm.attr("vox_center");
                            py::object py_vox_size  = py_svm.attr("vox_size");

                            at::Tensor octpath = py_octpath.cast<at::Tensor>().contiguous();     // [N,1] int64
                            at::Tensor L       = py_octlv.cast<at::Tensor>().contiguous();       // [N,1] int8 or int64
                            at::Tensor vox_center = py_vox_center.cast<at::Tensor>().contiguous(); // [N,3]
                            at::Tensor vox_size   = py_vox_size.cast<at::Tensor>().contiguous();   // [N,1] or [N]

                            // Basic sanity: same N
                            TORCH_CHECK(octpath.size(0) == N,
                                        "octpath.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(L.size(0) == N,
                                        "octlevel.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(vox_center.size(0) == N,
                                        "vox_center.size(0) != N in pruning visibility filter");
                            TORCH_CHECK(vox_center.size(1) == 3,
                                        "vox_center.size(1) must be 3");
                            if (vox_size.dim() == 1) {
                                vox_size = vox_size.view({N,1});
                            } else if (vox_size.dim() == 2) {
                                TORCH_CHECK(vox_size.size(0) == N,
                                            "vox_size.size(0) != N in pruning visibility filter");
                            } else {
                                TORCH_CHECK(false, "vox_size must be [N] or [N,1]");
                            }

                            // Build Python list of CUDA MiniCams
                            py::list py_cams;
                            py::object py_cuda = torch_mod.attr("device")("cuda");

                            auto move_attr_to_cuda_if_tensor =
                                [&](py::object& obj, const char* name){
                                    if (py::hasattr(obj, name)) {
                                        py::object t = obj.attr(name);
                                        if (py::hasattr(t, "is_cuda") &&
                                            !py::bool_(t.attr("is_cuda"))) {
                                            obj.attr(name) = t.attr("to")(py_cuda);
                                        }
                                    }
                                };

                            for (const auto& c : tr_cams) {
                                py::object py_cam = MiniCam_to_py(c);
                                move_attr_to_cuda_if_tensor(py_cam, "w2c");
                                move_attr_to_cuda_if_tensor(py_cam, "c2w");
                                move_attr_to_cuda_if_tensor(py_cam, "position");
                                move_attr_to_cuda_if_tensor(py_cam, "lookat");
                                py_cams.append(py_cam);
                            }

                            auto Nu_before = octpath.size(0);
                            TORCH_CHECK(Nu_before == N,
                                        "octpath.size(0) != N before visibility filter");

                            // 1) visibility: rate > 0
                            at::Tensor rate = svr_mod.attr("mark_max_samp_rate")(
                                py_cams,
                                py::cast(octpath),
                                py::cast(vox_center),
                                py::cast(vox_size)
                            ).cast<at::Tensor>();        // [N,1] or [N]

                            if (rate.dim() == 2 && rate.size(1) == 1)
                                rate = rate.squeeze(1);
                            rate = rate.to(torch::kFloat32);

                            at::Tensor kept = rate > 0.0f;   // [N] bool
                            int64_t n_rate_pos = kept.sum().item<int64_t>();

                            // 2) near filtering
                            const float near_thresh = 0.2f;  // same as createFromPcd
                            int64_t n_near_hit = 0;
                            if (near_thresh > 0.0f) {
                                at::Tensor is_near = svr_mod.attr("mark_near")(
                                    py_cams,
                                    py::cast(octpath),
                                    py::cast(vox_center),
                                    py::cast(vox_size),
                                    py::float_(near_thresh)
                                ).cast<at::Tensor>();       // [N,1] or [N]
                                if (is_near.dim() == 2 && is_near.size(1) == 1)
                                    is_near = is_near.squeeze(1);
                                is_near = is_near.to(torch::kBool);
                                kept    = kept & (~is_near);
                                n_near_hit = is_near.sum().item<int64_t>();

                                // --- DEBUG: save near voxels for rerun visualization ---
                                auto near_idx = is_near.nonzero().squeeze(1);  // [K]
                                if (near_idx.numel() > 0) {
                                    // note: vox_center, vox_size are the tensors from svm()
                                    debug_near_centers = vox_center.index({near_idx}).clone();  // [K,3]
                                    debug_near_sizes   = vox_size.index({near_idx}).clone();    // [K,1] or [K]
                                    debug_has_near     = true;

                                    std::cout << "[DEBUG NEAR] saved " << near_idx.size(0)
                                            << " near voxels for rerun visualization\n";
                                } else {
                                    debug_has_near = false;
                                }
                            }

                            kept = kept.view({-1}).to(torch::kBool);    // [N]
                            torch::Tensor prune_mask_vis = (~kept);     // [N]

                            // Combine with existing prune_mask
                            prune_mask = prune_mask | prune_mask_vis;

                            int64_t K = kept.sum().item<int64_t>();
                            std::cout << "[PRUNE/visibility] Nu_before=" << Nu_before
                                    << " rate>0=" << n_rate_pos
                                    << " near_hit=" << n_near_hit
                                    << " kept_final=" << K << std::endl;
                        } else {
                            std::cout << "[PRUNE/visibility] svm() is None, skipping visibility filter.\n";
                        }

                    } catch (const std::exception& e) {
                        std::cerr << "[PRUNE/visibility] exception: " << e.what() << "\n";
                    }
                }

                const int n_prune_final =
                    prune_mask.defined() ? (int)prune_mask.sum().item<int64_t>() : -1;
                std::cout << "[PRUNE:final] N=" << N
                        << " prune_sum=" << n_prune_final << "\n";

                const int n_can   = stat.max_w.size(0);
                const int n_prune = (int)(prune_mask.defined()? prune_mask.sum().item<int64_t>() : -1);
                std::cout << "[PRUNE:prep] thresh=" << prune_thres
                        << "  N=" << n_can
                        << "  prune_sum=" << n_prune << "\n";

                voxel_model_->pruning(prune_mask);
                const int new_n = voxel_model_->numVoxels();

                std::cout << "[PRUNING]     " << std::setw(7) << ori_n
                        << " => "          << std::setw(7) << new_n
                        << " (x" << std::fixed << std::setprecision(2)
                        << (double)new_n / std::max(1, ori_n)
                        << "; thres=" << std::setprecision(4) << prune_thres << ")\n";

                std::cout << "[PRUNE:after] N=" << new_n << "\n";
                auto pr_dbg = voxel_model_->subdivisionPriority();
                std::cout << "    priority def=" << (pr_dbg.defined()?1:0)
                        << " shape=" << shp(pr_dbg)
                        << " numel=" << (pr_dbg.defined()? pr_dbg.numel() : 0) << "\n";

                // If pruning changed the voxel set (or shapes don’t match), recompute stats
                const int M = voxel_model_->numVoxels();
                const bool shape_ok =
                    stat.min_samp_interval.defined() &&
                    stat.min_samp_interval.dim() == 2 &&
                    stat.min_samp_interval.size(0) == M &&
                    stat.min_samp_interval.size(1) == 1;
                if (new_n != ori_n || !shape_ok) {
                    stat = voxel_model_->computeTrainingStat(tr_cams);
                    std::cout << "[densify:stat-recompute] M=" << M
                            << "  max_w=" << shp(stat.max_w)
                            << "  min_itv=" << shp(stat.min_samp_interval)
                            << "  view_cnt=" << shp(stat.view_cnt) << "\n";
                }
            }

            // ---------------- SUBDIVIDE ----------------
            if (need_subdividing) {
                const int M = voxel_model_->numVoxels();
                if (M == 0) {
                    std::cout << "[SUBDIV:skip] M==0\n";
                } else {
                    // Align shapes
                    auto min_samp_interval = stat.min_samp_interval; // [M,1]
                    if (!min_samp_interval.defined() || min_samp_interval.size(0) != M) {
                        // last resort: recompute once more to align
                        stat = voxel_model_->computeTrainingStat(tr_cams);
                        min_samp_interval = stat.min_samp_interval;
                        std::cout << "[SUBDIV:stat-align] M=" << M
                                << "  min_itv=" << shp(min_samp_interval) << "\n";
                    }
                    if (min_samp_interval.dim() == 1)
                        min_samp_interval = min_samp_interval.view({M,1});

                    auto size_thres = min_samp_interval * opt_params_.subdivide_samp_thres_; // [M,1]

                    auto vox_size = voxel_model_->voxSize(); // [M] or [M,1]
                    if (vox_size.dim() == 1) vox_size = vox_size.view({M,1});
                    else if (vox_size.dim() == 2 && vox_size.size(1) == 1) { /* ok */ }
                    else vox_size = vox_size.reshape({M,1});

                    auto large_enough = (vox_size * 0.5 > size_thres).squeeze(1); // [M] bool

                    auto octlv = voxel_model_->octLevel(); // [M] or [M,1]
                    if (octlv.defined() && octlv.dim()==2 && octlv.size(1)==1) octlv = octlv.squeeze(1);
                    auto non_finest = (octlv.to(torch::kInt32) < voxel_model_->maxNumLevels()); // [M] bool

                    auto valid_mask = large_enough & non_finest; // [M] bool
                    const int64_t n_valid = valid_mask.defined()? valid_mask.sum().item<int64_t>() : 0;

                    // Priority: may be undefined/empty right after structural changes.
                    auto priority = voxel_model_->subdivisionPriority(); // should be 1-D [M]
                    if (!priority.defined() || priority.numel() != M) {
                        priority = torch::zeros(
                            {M},
                            torch::TensorOptions().dtype(torch::kFloat32).device(valid_mask.device()));
                    } else if (priority.dim() == 2 && priority.size(1) == 1) {
                        priority = priority.squeeze(1);
                    } else if (priority.dim() != 1) {
                        priority = priority.reshape({M});
                    }

                    std::cout << "[SUBDIV:prep] N=" << M
                            << "  large_enough=" << shp(large_enough)
                            << "  octlevel="    << shp(voxel_model_->octLevel())
                            << "  octlv(1D)="   << shp(octlv)
                            << "  non_finest="  << shp(non_finest)
                            << "  valid_mask="  << shp(valid_mask)
                            << "  sum(valid)="  << n_valid << "\n";
                    std::cout << "    priority def=" << (priority.defined()?1:0)
                            << " shape=" << shp(priority)
                            << " numel=" << priority.numel() << "\n";
                    std::cout << "    CHECK: priority.size(0)="
                            << (priority.defined() && priority.dim()? priority.size(0) : -1)
                            << "  vs valid_mask.size(0)="
                            << (valid_mask.defined()? valid_mask.size(0) : -1) << "\n";

                    if (n_valid == 0) {
                        std::cout << "[SUBDIV:skip] n_valid=0\n";
                    } else {
                        // mask to zero for invalids (keeps length M)
                        priority = priority * valid_mask.to(priority.scalar_type());

                        torch::Tensor subdivide_mask;
                        if (iter <= opt_params_.subdivide_all_until_) {
                            subdivide_mask = valid_mask; // take all valids early on
                        } else {
                            // compute quantile on valid entries only
                            auto pos_idx  = valid_mask.nonzero().squeeze(1);     // [K]
                            auto pos_vals = priority.index({pos_idx});            // [K]
                            double q = std::max(0.0, 1.0 - (double)opt_params_.subdivide_prop_);
                            auto thres = (pos_vals.numel() > 0)
                                    ? pos_vals.quantile(q)
                                    : torch::tensor(0.0, pos_vals.options());
                            subdivide_mask = torch::zeros_like(valid_mask, torch::kBool);
                            if (pos_vals.numel() > 0) {
                                auto pick = (pos_vals > thres);                    // [K]
                                subdivide_mask.index_put_({pos_idx}, pick);
                                subdivide_mask = subdivide_mask & valid_mask;      // just in case
                            }
                        }

                        // cap number of parents (each makes +7 children)
                        int max_n_subdiv = std::round(
                            (opt_params_.subdivide_max_num_ - voxel_model_->numVoxels()) / 7.0);

                        if (max_n_subdiv > 0) {
                            int num_sel = (int)subdivide_mask.sum().item<int64_t>();
                            if (num_sel > max_n_subdiv) {
                                auto pos_idx  = subdivide_mask.nonzero().squeeze(1); // [K]
                                auto pos_vals = priority.index({pos_idx});           // [K]
                                auto sorted   = std::get<0>(pos_vals.sort(/*dim=*/0)); // asc
                                int  n_removed = num_sel - max_n_subdiv;
                                auto cutoff    = sorted.index({n_removed - 1});
                                subdivide_mask = subdivide_mask & (priority > cutoff);
                            }

                            const int before = voxel_model_->numVoxels();
                            voxel_model_->subdividing(subdivide_mask);
                            const int after  = voxel_model_->numVoxels();
                            std::cout << "[SUBDIVIDING] " << std::setw(7) << before
                                    << " => "          << std::setw(7) << after
                                    << " (x" << std::fixed << std::setprecision(2)
                                    << (double)after / std::max(1, before) << ")\n";

                            voxel_model_->resetSubdivisionPriority();
                        } else {
                            std::cout << "[SUBDIV:skip] cap reached (max_n_subdiv<=0)\n";
                        }
                    }
                }
            }
            voxel_model_->createTrainer(
                opt_params_.geo_lr_,
                opt_params_.sh0_lr_,
                opt_params_.shs_lr_,
                opt_params_.optim_beta1_,
                opt_params_.optim_beta2_,
                opt_params_.optim_eps_,
                opt_params_.lr_decay_ckpt_,
                opt_params_.lr_decay_mult_
            );
            voxel_model_->schedulerLoadStateDict(sched_state);
            // Empty CUDA cache as SV does
            {
                py::gil_scoped_acquire gil;
                py::module_ torch_mod = py::module_::import("torch");
                torch_mod.attr("cuda").attr("empty_cache")();
            }
            last_densify_iter_ = iter;
        }
    }
    // Update learning rate
    voxel_model_->schedulerStep();

        // ----- 1) FULL VOXELS (unchanged) -----
    torch::Tensor centers_all = voxel_model_->voxCenter(); // [N,3]
    torch::Tensor sizes_all   = voxel_model_->voxSize();   // [N] or [N,1]
    // colors from SH0 + density as before
    torch::Tensor colors_all;
    {
        torch::Tensor sh0 = voxel_model_->sh0();
        {
            py::gil_scoped_acquire gil2;
            static py::module act_mod = py::module::import("src.utils.activation_utils");
            py::object rgb_py = act_mod.attr("shzero2rgb")(py::cast(sh0));
            colors_all = rgb_py.cast<torch::Tensor>().contiguous();
        }

        torch::Tensor density = voxel_model_->voxelDensityMean();
        if (density.defined() && density.numel() == centers_all.size(0)) {
            auto d_cpu = density.view({-1}).to(torch::kCPU);
            float d_min = d_cpu.min().item().toFloat();
            float d_max = d_cpu.max().item().toFloat();
            float eps   = 1e-6f;
            float range = d_max - d_min;

            torch::Tensor alpha_cpu;
            if (range < eps) {
                alpha_cpu = torch::full_like(d_cpu, 0.8f);
            } else {
                alpha_cpu = (d_cpu - d_min) / range;
                alpha_cpu = alpha_cpu.clamp(0.05f, 1.0f);
            }
            auto col_cpu = colors_all.to(torch::kCPU);
            TORCH_CHECK(col_cpu.dim() == 2 &&
                        col_cpu.size(0) == alpha_cpu.size(0),
                        "colors and density must have same N");
            if (col_cpu.size(1) == 3) {
                auto N = col_cpu.size(0);
                auto col_rgba = torch::zeros({N, 4}, col_cpu.options());
                col_rgba.index_put_(
                    {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                    col_cpu
                );
                col_rgba.index_put_(
                    {torch::indexing::Slice(), 3},
                    alpha_cpu
                );
                colors_all = col_rgba.to(colors_all.device());
            } else if (col_cpu.size(1) == 4) {
                col_cpu.index_put_(
                    {torch::indexing::Slice(), 3},
                    alpha_cpu
                );
                colors_all = col_cpu.to(colors_all.device());
            } else {
                TORCH_CHECK(false, "colors must be [N,3] or [N,4]");
            }
        }
    }
    // visualize full field (same as before)
    sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
        centers_all, sizes_all, colors_all, iter
    );
    // ----- 2) NEAR VOXELS (debug overlay) -----
    if (debug_has_near &&
        debug_near_centers.defined() &&
        debug_near_centers.numel() > 0)
    {
        auto centers_near = debug_near_centers;         // [K,3] CUDA or CPU
        auto sizes_near   = debug_near_sizes;           // [K,1] or [K]

        // ensure sizes_near is [K,1] on CPU
        if (sizes_near.dim() == 1) {
            sizes_near = sizes_near.view({sizes_near.size(0), 1});
        } else if (sizes_near.dim() == 2 && sizes_near.size(1) == 1) {
            // ok
        } else {
            sizes_near = sizes_near.reshape({sizes_near.size(0), 1});
        }

        auto K = centers_near.size(0);
        torch::Tensor colors_near = torch::zeros({K, 4}, centers_near.options());
        colors_near.index_put_({torch::indexing::Slice(), 0}, 1.0f);  // R
        colors_near.index_put_({torch::indexing::Slice(), 3}, 0.7f);  // alpha

        std::cout << "[DEBUG NEAR] visualizing " << K
                << " near voxels in rerun (red boxes)\n";

        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            centers_near,
            sizes_near,
            colors_near,
            iter,
            "world/voxels_near"    // <-- separate entity in blueprint
        );
    }
    // ----- 3) TSDF-PRUNED VOXELS (debug overlay) -----
    if (debug_has_tsdf &&
        debug_tsdf_centers.defined() &&
        debug_tsdf_centers.numel() > 0)
    {
        auto centers_tsdf = debug_tsdf_centers;   // [K_tsdf,3]
        auto sizes_tsdf   = debug_tsdf_sizes;     // [K_tsdf,1] or [K_tsdf]

        // ensure sizes_tsdf is [K_tsdf,1]
        if (sizes_tsdf.dim() == 1) {
            sizes_tsdf = sizes_tsdf.view({sizes_tsdf.size(0), 1});
        } else if (sizes_tsdf.dim() == 2 && sizes_tsdf.size(1) == 1) {
            // ok
        } else {
            sizes_tsdf = sizes_tsdf.reshape({sizes_tsdf.size(0), 1});
        }

        auto Kt = centers_tsdf.size(0);
        // visualize with a different color, e.g. blue
        torch::Tensor colors_tsdf = torch::zeros({Kt, 4}, centers_tsdf.options());
        colors_tsdf.index_put_({torch::indexing::Slice(), 2}, 1.0f);  // B = 1
        colors_tsdf.index_put_({torch::indexing::Slice(), 3}, 0.7f);  // alpha = 0.7

        std::cout << "[DEBUG TSDF] visualizing " << Kt
                << " TSDF-pruned voxels in rerun (blue boxes)\n";

        sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
            centers_tsdf,
            sizes_tsdf,
            colors_tsdf,
            iter,
            "world/voxels_tsdf"   // <-- separate entity path
        );
    }
    // torch::Tensor centers = voxel_model_->voxCenter(); // [N,3], device = CUDA
    // torch::Tensor sizes   = voxel_model_->voxSize();   // [N] or [N,1]
    // // Simple debug color: map octlevel to a colormap, or use density / tsdf.
    // torch::Tensor colors; // start with undefined → no color
    // torch::Tensor sh0 = voxel_model_->sh0();
    // py::gil_scoped_acquire gil;
    // static py::module act_mod = py::module::import("src.utils.activation_utils");
    // // act_utils.shzero2rgb(sh0) -> [N,3], float in [0,1]
    // py::object rgb_py = act_mod.attr("shzero2rgb")(py::cast(sh0));
    // colors = rgb_py.cast<torch::Tensor>().contiguous();

    // torch::Tensor density = voxel_model_->voxelDensityMean();  // [Nv]
    // if (density.defined() && density.numel() == centers.size(0)) {
    //     // flatten and move to CPU for simple scalar ops
    //     auto d_cpu = density.view({-1}).to(torch::kCPU); 

    //     // Use non-templated item() to avoid macro/template clashes
    //     float d_min = d_cpu.min().item().toFloat();
    //     float d_max = d_cpu.max().item().toFloat();
    //     float eps   = 1e-6f;
    //     float range = d_max - d_min;

    //     torch::Tensor alpha_cpu;
    //     if (range < eps) {
    //         // degenerate case: all same → constant semi-opaque
    //         alpha_cpu = torch::full_like(d_cpu, 0.8f);  // in [0,1]
    //     } else {
    //         alpha_cpu = (d_cpu - d_min) / range;        // [0,1]
    //         // avoid fully invisible voxels, keep a minimum opacity
    //         alpha_cpu = alpha_cpu.clamp(0.05f, 1.0f);
    //     }

    //     // bring colors to CPU to combine with alpha
    //     auto col_cpu = colors.to(torch::kCPU);         
    //     TORCH_CHECK(col_cpu.dim() == 2 &&
    //                 col_cpu.size(0) == alpha_cpu.size(0),
    //                 "colors and density must have same N");

    //     if (col_cpu.size(1) == 3) {
    //         // RGB -> RGBA
    //         auto N = col_cpu.size(0);
    //         auto col_rgba = torch::zeros({N, 4}, col_cpu.options());
    //         // copy RGB
    //         col_rgba.index_put_(
    //             {torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
    //             col_cpu
    //         );
    //         // write alpha in [0,1]
    //         col_rgba.index_put_(
    //             {torch::indexing::Slice(), 3},
    //             alpha_cpu
    //         );
    //         colors = col_rgba.to(colors.device());  // back to original device
    //     } else if (col_cpu.size(1) == 4) {
    //         // already RGBA → overwrite alpha with our mapping
    //         col_cpu.index_put_(
    //             {torch::indexing::Slice(), 3},
    //             alpha_cpu
    //         );
    //         colors = col_cpu.to(colors.device());
    //     } else {
    //         TORCH_CHECK(false, "colors must be [N,3] or [N,4]");
    //     }
    // }
    // sv::RerunVisualizerBridge::instance().visualizeVoxelBoxes(
    //     centers, sizes, colors, iter
    // );
    // sv::RerunVisualizerBridge::instance().visualizeSVRasterMesh(
    // centers, sizes, colors, iter);

    // --- Incremental TSDF mesh visualization (NVBlox-style) ---
    const int tsdf_vis_interval = 100;  // e.g., visualize every 50 iters
    if (sensor_type_ == RGBD && use_tsdf_mapping_ && (iter % tsdf_vis_interval == 0)) {
        try {
            // 1) Update the mesh from the TSDF layer
            sdf_mapper_->updateColorMesh();

            // 2) Dump to PLY
            const auto tsdf_dir = result_dir_ / "tsdf_mesh";
            std::filesystem::create_directories(tsdf_dir);
            const auto ply_path =
                (tsdf_dir / ("tsdf_mesh_iter_" + std::to_string(iter) + ".ply")).string();

            nvblox::io::outputColorMeshLayerToPly(
                sdf_mapper_->color_mesh_layer(),
                ply_path
            );
            // 3) Ask Rerun to show this mesh
            sv::RerunVisualizerBridge::instance()
                .visualizeNvbloxPlyMesh(ply_path, iter);
        }
        catch (const std::exception& e) {
            std::cerr << "[TSDF/RERUN] exception in incremental TSDF mesh viz: "
                      << e.what() << "\n";
        }
    }

    if (mDevice == torch::kCUDA) torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item<float>() + 0.6f * ema_loss_for_log_;

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(
                masked_image,
                gt_image,
                viewpoint_cam->fid_,
                result_dir_, result_dir_, result_dir_
            );
        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

         // Log and save
         if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
             sv::VoxelTrainer::trainingReport(
                 getIteration(),
                 opt_params_.iterations_,
                 Ll1,
                 loss,
                 ema_loss_for_log_,
                 mse,
                 iter_time,
                 *voxel_model_,
                 *scene_,
                 pipe_params_,
                 background_
             );

        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(iteration_) / "ply");
        }
        
        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Extract scalars for csv logging
        const float l1_val    = Ll1.item<float>();
        const float l2_val    = mse.item<float>();
        const float photoslam_val = photoslam_loss.item<float>();
        // Combined CSV (iter, l1, l2, train_loss)
        if (loss_log_) {
            loss_log_ << std::fixed << std::setprecision(6)
                    << iter << ',' << l1_val << ',' << l2_val << ',' << photoslam_val << '\n';
        }
        if (loss_l1_log_) { loss_l1_log_ << std::fixed << std::setprecision(6) << iter << ',' << l1_val << '\n'; }
        if (loss_l2_log_) { loss_l2_log_ << std::fixed << std::setprecision(6) << iter << ',' << l2_val << '\n'; }
        if (loss_ssim_log_) { loss_ssim_log_ << std::fixed << std::setprecision(6) << iter << ',' << photoslam_val << '\n'; }
        if ((iter % 50) == 0) {
            loss_log_.flush();
            loss_l1_log_.flush();
            loss_l2_log_.flush();
            loss_ssim_log_.flush();
        }

        float loss_val = loss.item<float>();
        int fid = viewpoint_cam->fid_;                 // key-frame ID being trained
        // --- 1) ensure our vectors are big enough ---
        if (fid >= static_cast<int>(best_loss_per_kf_.size())) {
            size_t newSize = fid + 1;
            best_loss_per_kf_.resize(newSize,
                                    std::numeric_limits<float>::infinity());
            worst_loss_per_kf_.resize(newSize,
                                    -std::numeric_limits<float>::infinity());
        }
        // references into the right slot
        float &best  = best_loss_per_kf_[fid];
        float &worst = worst_loss_per_kf_[fid];
        // --- 2) update “best” for this KF ---
        if (loss_val < best) {
            best = loss_val;
            auto kf_dir = extrema_dir_ / ("kf" + std::to_string(fid));
            std::filesystem::create_directories(kf_dir);
            saveTensor(gt_image,     "best_gt",     kf_dir.string(), iteration_, fid);
            saveTensor(masked_image, "best_masked", kf_dir.string(), iteration_, fid);
        }
        // --- 3) update “worst” for this KF ---
        if (loss_val > worst) {
            worst = loss_val;
            auto kf_dir = extrema_dir_ / ("kf" + std::to_string(fid));
            std::filesystem::create_directories(kf_dir);
            saveTensor(gt_image,     "worst_gt",     kf_dir.string(), iteration_, fid);
            saveTensor(masked_image, "worst_masked", kf_dir.string(), iteration_, fid);
        }
    }
}

void VoxelMapper::combineMappingOperations()
{
    // Get Mapping Operations
    while (mpSLAM->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            mpSLAM->getAtlas()->getAndPopMappingOperation();
        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
        bool kf_changed = false;
            // std::cout << "[Gaussian Mapper]Local BA Detected."
            //           << std::endl;
            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.
                // Otherwise create a new one
                if (pkf) {
                //  std::cout << "if pkf" << std::endl;
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                //  pkf->computeTransformTensors();
                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                    kf_changed = true;
                }
                else {
                // std::cout << "no pkf" << std::endl;
                handleNewKeyframe(kf);                   // still void
                if (pkf) {
                    // Its original_image_ is Float32 RGB in [0..1], shape (3,H,W)
                    torch::Tensor chw = pkf->original_image_.detach().cpu().clamp(0,1);
                    torch::Tensor hwc = chw.permute({1,2,0}).contiguous(); // (H,W,3), float32
                    // Copy into a CV_32FC3 Mat (RGB)
                    const int H = hwc.size(0);
                    const int W = hwc.size(1);
                    cv::Mat img_float(H, W, CV_32FC3);
                    std::memcpy(img_float.data, hwc.data_ptr<float>(), H*W*3*sizeof(float));
                    static const auto proj_dir = result_dir_ / "proj_debug";
                    static const auto imgs_dir = proj_dir / "imgs";
                    std::filesystem::create_directories(imgs_dir);
                    saveKfPng_fromFloatRGB(img_float, kfid, imgs_dir);
                }
                }
            }
            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            const int iter = getIteration();
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
            // log_increase_batch_npy(result_dir_, points, colors, getIteration(), next_batch_index_);
            // ++next_batch_index_;
            //  voxel_model_->increasePcd(points, colors, getIteration(), kfs_for_bounding);
                // py::object sched_state = voxel_model_->schedulerStateDict();

                // Build training camera list from the keyframes we keep in the scene
                std::vector<sv::MiniCam> tr_cams;
                tr_cams.reserve(scene_->keyframes().size());
                for (auto& kv : scene_->keyframes()) {
                    if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
                }
                voxel_model_->increasePcd(points, colors, getIteration(), tr_cams);
                if (voxel_model_ && voxel_model_->consumeArtifactFillFlag()) {
                    last_artifact_fill_iter_ = static_cast<int64_t>(iter);
                    std::cout << "[VoxelMapper] artifact fill happened at iter "
                            << iter << "\n";
                }
                // voxel_model_->createTrainer(
                //                             opt_params_.geo_lr_,
                //                             opt_params_.sh0_lr_,
                //                             opt_params_.shs_lr_,
                //                             opt_params_.optim_beta1_,
                //                             opt_params_.optim_beta2_,
                //                             opt_params_.optim_eps_,
                //                             opt_params_.lr_decay_ckpt_,
                //                             opt_params_.lr_decay_mult_);
                // voxel_model_->schedulerLoadStateDict(sched_state);
            }

            // // Only try to grow if model was initialized and we have any new points
            // // if (initial_mapped_ && !points.empty()) {
            // if (initial_mapped_ && points.size() >= 30) {
            //     // 1) Get current bound
            //     torch::Tensor sc  = voxel_model_->SceneCenter();  // [3] if set
            //     torch::Tensor se  = voxel_model_->SceneExtent();  // [1] if set
            //     // If the model hasn't been initialized yet (shouldn’t happen here), skip.
            //     if (sc.defined() && se.defined() && sc.numel()==3 && se.numel()==1) {
            //         auto sc_cpu = sc.detach().to(torch::kCPU);
            //         auto se_cpu = se.detach().to(torch::kCPU);
            //         // scene_min = center - 0.5 * extent;  scene_max = center + 0.5 * extent
            //         float cx = sc_cpu[0].item<float>();
            //         float cy = sc_cpu[1].item<float>();
            //         float cz = sc_cpu[2].item<float>();
            //         float ex = se_cpu[0].item<float>() * 0.5f;
            //         float minx = cx - ex, maxx = cx + ex;
            //         float miny = cy - ex, maxy = cy + ex;
            //         float minz = cz - ex, maxz = cz + ex;
            //         const float eps = 1e-6f; // tiny numerical slack
            //         // 2) Keep only points outside current AABB
            //         std::vector<float> out_pts;
            //         std::vector<float> out_cols;
            //         out_pts.reserve(points.size());  // upper bound
            //         out_cols.reserve(colors.size());
            //         const size_t N = points.size() / 3;
            //         for (size_t i = 0; i < N; ++i) {
            //             float x = points[3*i + 0];
            //             float y = points[3*i + 1];
            //             float z = points[3*i + 2];
            //             bool outside =
            //                 (x < minx - eps) || (x > maxx + eps) ||
            //                 (y < miny - eps) || (y > maxy + eps) ||
            //                 (z < minz - eps) || (z > maxz + eps);
            //             if (outside) {
            //                 out_pts.push_back(x);
            //                 out_pts.push_back(y);
            //                 out_pts.push_back(z);

            //                 out_cols.push_back(colors[3*i + 0]);
            //                 out_cols.push_back(colors[3*i + 1]);
            //                 out_cols.push_back(colors[3*i + 2]);
            //             }
            //         }
            //         // 3) Only grow if enough out-of-bounds points
            //         const int MIN_OUTSIDE = 10; // your threshold
            //         if ((int)(out_pts.size()/3) >= MIN_OUTSIDE) {
            //             // Optional: ensure we don't move the 'min side' if you want perfect key stability
            //             // (Counts of which side is violated; skip if it would force min to move)
            //             // bool touches_min_side = (min of any coord < minX/Y/Z);
            //             // if (touches_min_side) { /* decide: defer / tile / rebase */ }
            //             torch::NoGradGuard no_grad;
            //             std::unique_lock<std::mutex> lock_render(mutex_render_);
            //             // 4) Actually insert *only* the outside points
            //             voxel_model_->increasePcd(out_pts, out_cols, getIteration());
            //             // voxel_model_->increasePcd(points, colors, getIteration());
            //         }
            //     }
            // }
            // if (kf_changed) {
            //     dumpKeyframesForProjectionFile(scene_->keyframes(), scene_->cameras_,
            //                                 result_dir_ / "proj_debug");
            // }
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cout << "[Voxel Mapper]Loop Closure Detected."
                    << std::endl;

            bool kf_changed = false;
            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();

            // std::vector<std::shared_ptr<VoxelKeyframe>> kfs_for_bounding;

             // Mark the transformed points to avoid transforming more than once
             torch::Tensor point_not_transformed_flags =
                 torch::full(
                     {voxel_model_->center_.size(0)},
                     true,
                     torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            //  if (record_loop_ply_)
            //      savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
             int num_transformed = 0;
             // Add keyframes to the scene
             for (auto& kf : associated_kfs) {
                 // Keyframe Id
                 auto kfid = std::get<0>(kf);
                 std::shared_ptr<VoxelKeyframe> pkf = scene_->getKeyframe(kfid);
                 // In case new points are added in handleNewKeyframe()
                 int64_t num_new_points = voxel_model_->center_.size(0) - point_not_transformed_flags.size(0);
                 if (num_new_points > 0)
                     point_not_transformed_flags = torch::cat({
                         point_not_transformed_flags,
                         torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                         /*dim=*/0);
                 // If kf is already in the scene, evaluate the change in pose,
                 // if too large we perform loop correction on its visible model points.
                 // If not in the scene, create a new one.
                 if (pkf) {
                     auto& pose = std::get<2>(kf);
                     // If is loop closure kf
 // if (std::get<4>(kf)) {
 // renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                         Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                         Sophus::SE3f inv_pose = pose.inverse();
                         Sophus::SE3f diff_pose = inv_pose * original_pose;
                         bool large_rot = !diff_pose.rotationMatrix().isApprox(
                             Eigen::Matrix3f::Identity(), large_rot_th_);
                         bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                             1.0, large_trans_th_);
                         if (large_rot || large_trans) {
                             std::cout << "[Voxel Mapper]Large loop correction detected, transforming visible points of kf "
                                     << kfid << std::endl;
                             diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                             diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                             diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                             torch::Tensor diff_pose_tensor =
                                 tensor_utils::EigenMatrix2TorchTensor(
                                     diff_pose.matrix(), device_type_).transpose(0, 1);
                            //  {
                            //      std::unique_lock<std::mutex> lock_render(mutex_render_);
                            //      voxel_model_->scaledTransformVisiblePointsOfKeyframe(
                            //          point_not_transformed_flags,
                            //          diff_pose_tensor,
                            //          pkf->world_view_transform_,
                            //          pkf->full_proj_transform_,
                            //          pkf->creation_iter_,
                            //          stableNumIterExistence(),
                            //          num_transformed,
                            //          loop_kf_scale); // selected xyz *= s
                            //  }
                             // Give loop keyframes times of use
                             increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
 // renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
 // std::cout<<num_transformed<<std::endl;
                         }
 // }
                     pkf->setPose(
                         pose.unit_quaternion().cast<double>(),
                         pose.translation().cast<double>());
                    //  pkf->computeTransformTensors();
 // if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");

                    kf_changed = true;
                 }
                 else {
                    //  std::cout << "no pkf again" << std::endl;
                     handleNewKeyframe(kf);
                 }
             }
            //  if (record_loop_ply_)
            //      savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
 // keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
 
             // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
             auto& associated_points = opr.associatedMapPoints();
             auto& points = std::get<0>(associated_points);
             auto& colors = std::get<1>(associated_points);

             // Add new points to the model
             if (initial_mapped_ && points.size() >= 30) {
                std::cout << "adds new points" << std::endl;
                extendAABB_with_flat_xyz(aabb_min_, aabb_max_, points);  // points = std::vector<float>
                have_bounds_ = true;

                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                // voxel_model_->increasePcd(points, colors, getIteration(), kfs_for_bounding);
                // py::object sched_state = voxel_model_->schedulerStateDict();
                // voxel_model_->increasePcd(points, colors, getIteration());
                // voxel_model_->createTrainer(
                //                             opt_params_.geo_lr_,
                //                             opt_params_.sh0_lr_,
                //                             opt_params_.shs_lr_,
                //                             opt_params_.optim_beta1_,
                //                             opt_params_.optim_beta2_,
                //                             opt_params_.optim_eps_,
                //                             opt_params_.lr_decay_ckpt_,
                //                             opt_params_.lr_decay_mult_);
                // voxel_model_->schedulerLoadStateDict(sched_state);
             }
 
            // Mark this iteration
            loop_closure_iteration_ = true;
            // if (kf_changed) {
            //     dumpKeyframesForProjectionFile(scene_->keyframes(), scene_->cameras_,
            //                                 result_dir_ / "proj_debug");
            // }
         }
         break;
 
         case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
         {
             std::cout << "[Voxel Mapper]Scale refinement Detected. Transforming all kfs and points..."
                       << std::endl;
 
             float s = opr.mfScale;
             Sophus::SE3f& T = opr.mT;
             if (initial_mapped_) {
                 // Apply the scaled transformation on gaussian model points
                 {
                     std::unique_lock<std::mutex> lock_render(mutex_render_);
                    //  voxel_model_->applyScaledTransformation(s, T);
                 }
                 // Apply the scaled transformation to the scene
                //  scene_->applyScaledTransformation(s, T);
             }
             else { // TODO: the workflow should not come here, delete this branch
                 // Apply the scaled transformation to the cached points
                 for (auto& pt : scene_->cached_point_cloud_) {
                     // pt <- (s * Ryw * pt + tyw)
                     auto& pt_xyz = pt.second.xyz_;
                     pt_xyz *= s;
                     pt_xyz = T.cast<double>() * pt_xyz;
                 }
 
                 // Apply the scaled transformation on gaussian keyframes
                 for (auto& kfit : scene_->keyframes()) {
                     std::shared_ptr<VoxelKeyframe> pkf = kfit.second;
                     Sophus::SE3f Twc = pkf->getPosef().inverse();
                     Twc.translation() *= s;
                     Sophus::SE3f Tyc = T * Twc;
                     Sophus::SE3f Tcy = Tyc.inverse();
                     std::cout << "ScaleRefinement: kf " << Tcy.translation() << std::endl;
                     pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    //  pkf->computeTransformTensors();
                 }
             }
         }
         break;
 
         default:
         {
             throw std::runtime_error("MappingOperation type not supported!");
         }
         break;
         }
     }
 }

 bool VoxelMapper::hasMetInitialMappingConditions() {
     if (!mpSLAM->isShutDown() &&
         mpSLAM->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
         mpSLAM->getAtlas()->hasMappingOperation())
         return true;
 
     bool conditions_met = false;
     return conditions_met;
}

bool VoxelMapper::hasMetIncrementalMappingConditions() {
     if (!mpSLAM->isShutDown() &&
         mpSLAM->getAtlas()->hasMappingOperation())
         return true;
 
     bool conditions_met = false;
     return conditions_met;
}

void VoxelMapper::generateKfidRandomShuffle()
{
     if (scene_->keyframes().empty())
         return;
 
     std::size_t nkfs = scene_->keyframes().size();
     kfid_shuffle_.resize(nkfs);
     std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
     std::mt19937 g(rd_());
     std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);
 
     kfid_shuffled_ = true;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomSlidingWindowKeyframe()
{
    // If no keyframes, return nullptr
    if (scene_->keyframes().empty())
        return nullptr;

    // If not shuffled yet, build shuffle
    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<VoxelKeyframe> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

    return viewpoint_cam;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomKeyframe()
 {
     if (scene_->keyframes().empty())
         return nullptr;
 
     // Get randomly
     int nkfs = static_cast<int>(scene_->keyframes().size());
     int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
     auto random_cam_it = scene_->keyframes().begin();
     for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
         ++random_cam_it;
     std::shared_ptr<VoxelKeyframe> viewpoint_cam = (*random_cam_it).second;
 
     // Count used times
     auto viewpoint_fid = viewpoint_cam->fid_;
     if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
         kfs_used_times_[viewpoint_fid] = 1;
     else
         ++kfs_used_times_[viewpoint_fid];
 
     return viewpoint_cam;
 }

void VoxelMapper::cullKeyframes()
{
    // Ask ORB-SLAM3 which keyframe IDs are still “live”
    std::unordered_set<unsigned long> kfids =
        mpSLAM->getAtlas()->GetCurrentKeyFrameIds();

     std::vector<unsigned long> kfids_to_erase;
     std::size_t nkfs = scene_->keyframes().size();
     kfids_to_erase.reserve(nkfs);
     for (auto& kfit : scene_->keyframes()) {
         unsigned long kfid = kfit.first;
         if (kfids.find(kfid) == kfids.end()) {
             kfids_to_erase.emplace_back(kfid);
         }
     }
 
     for (auto& kfid : kfids_to_erase) {
         scene_->keyframes().erase(kfid);
     }
 }

void VoxelMapper::handleNewKeyframe(
    std::tuple<
        unsigned long,    // 0: keyframe ID
        unsigned long,    // 1: camera ID
        Sophus::SE3f,     // 2: pose
        cv::Mat,          // 3: RGB image
        bool,             // 4: loop‐closure flag (unused here)
        cv::Mat,          // 5: auxiliary (unused here)
        std::vector<float>, // 6: keypoint pixel coords (unused here)
        std::vector<float>, // 7: keypoint local coords (unused here)
        std::string> &kf       // 8: image filename (relative or absolute)
)
{
    // ─── Create a new VoxelKeyframe, exactly like Photo-SLAM’s Gaussian case ─
    std::shared_ptr<VoxelKeyframe> pkf  = std::make_shared<VoxelKeyframe>(std::get<0>(kf), getIteration());
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>()
    );
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    // Camera
    sv::Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    pkf->setCameraParams(camera);

    // Image (left if STEREO)
    cv::Mat imgRGB = std::get<3>(kf);
    if (this->sensor_type_ == STEREO)
        imgRGB_undistorted = imgRGB;
    else
        camera.undistortImage(imgRGB, imgRGB_undistorted);
    // Auxiliary Image
    cv::Mat imgAux = std::get<5>(kf);
    if (this->sensor_type_ == RGBD)
        camera.undistortImage(imgAux, imgAux_undistorted);
    else
        imgAux_undistorted = imgAux;

    pkf->original_image_ =
        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
    pkf->img_filename_ = std::get<8>(kf);
    pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
    pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
    pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
     
    // Add the new keyframe to the scene
    // pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;

    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }

    try {
        const unsigned long kf_id = std::get<0>(kf);
        // Build MiniCam at native resolution
        const int image_height = pkf->image_height_;
        const int image_width  = pkf->image_width_;
        sv::MiniCam cam = pkf->toMiniCam(image_height, image_width);        // cam.c2w is a 4x4 torch::Tensor, typically on CUDA; move to CPU.
        torch::Tensor c2w_cpu = cam.c2w.to(torch::kCPU).contiguous();
        TORCH_CHECK(c2w_cpu.sizes() == torch::IntArrayRef({4, 4}),
                    "MiniCam.c2w must be 4x4");
        // Torch is row-major; Eigen is column-major by default.
        // Map the data as a row-major Eigen matrix and then copy it into a normal Matrix4f.
        Eigen::Matrix4f T_W_C;
        {
            float* data = c2w_cpu.data_ptr<float>();
            Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> T_row_major(data);
            T_W_C = T_row_major;
        }
        // Now t and R are correct
        Eigen::Vector3f t = T_W_C.block<3,1>(0, 3);
        Eigen::Matrix3f R = T_W_C.block<3,3>(0, 0);
        Eigen::Quaternionf q(R);

        // Tracking image: use the undistorted RGB/BGR image of the keyframe.
        const cv::Mat& track_img = pkf->img_undist_;

        // Intrinsics for Rerun
        const float fx = static_cast<float>(camera.fx());
        const float fy = static_cast<float>(camera.fy());
        const float cx = static_cast<float>(camera.cx());
        const float cy = static_cast<float>(camera.cy());

        // For now, don't send any 2D keypoints (only pose + image).
        std::vector<Eigen::Vector2f> kps_uv;
        std::vector<int>             track_ids;

        sv::RerunVisualizerBridge::instance().visualizeCamera(
            T_W_C,
            track_img,
            std::vector<Eigen::Vector2f>{},
            std::vector<int>{},
            static_cast<int>(kf_id),
            fx, fy, cx, cy
        );
    } catch (const c10::Error& e) {
        std::cerr << "[RERUN] Torch error in visualizeCamera: "
                  << e.msg() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[RERUN] Exception in visualizeCamera: "
                  << e.what() << std::endl;
    }

    // // ─── nvblox: integrate this new keyframe into TSDF ───
    if (sensor_type_ == RGBD && use_tsdf_mapping_) {
        cv::Mat depth_meters;
        if (pkf->img_auxiliary_undist_.type() == CV_32FC1) {
            depth_meters = pkf->img_auxiliary_undist_;
        } else if (pkf->img_auxiliary_undist_.type() == CV_16UC1) {
            pkf->img_auxiliary_undist_.convertTo(depth_meters, CV_32FC1, 1.0 / 1000.0);
        } else {
            pkf->img_auxiliary_undist_.convertTo(depth_meters, CV_32FC1);
        }
        debugDepthStats(depth_meters, static_cast<int>(std::get<0>(kf)));
        integrateKeyframeIntoNvblox(*pkf, depth_meters);
    }
}

// We use torch::Tensor in the signature. The mangling is the same as at::Tensor
// because at::Tensor is an alias in libtorch.
// -----------------------------------------------------------------------------
// 1) Monocular inactive geo densify helper
//    For now: stub that returns 0 new points/colors.
//    This keeps the linker happy and is harmless in your RGBD experiments.
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor>
monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
    torch::Tensor& kps_pixel_tensor,      // [N,2]
    torch::Tensor& kps_has3D_tensor,      // [N] (bool)
    torch::Tensor& kps_point_local_tensor,// [N,3]
    torch::Tensor& colors,                // [H*W,3] float32 RGB in [0,1] or [0,255]
    float max_pixel_dist,
    std::vector<float>& intr,             // fx, fy, cx, cy
    int image_width)
{
    // Stub: do nothing, return empty tensors.
    // If you later want monocular inactive geo densify,
    // we can port the full Photo-SLAM implementation here.
    auto device = kps_pixel_tensor.device();
    auto opts   = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    torch::Tensor empty_pts   = torch::empty({0, 3}, opts);
    torch::Tensor empty_colors= torch::empty({0, 3}, opts);

    (void)kps_has3D_tensor;
    (void)kps_point_local_tensor;
    (void)colors;
    (void)max_pixel_dist;
    (void)intr;
    (void)image_width;

    return std::make_tuple(empty_pts, empty_colors);
}

// -----------------------------------------------------------------------------
// 2) Transform 3D points by a 4x4 pose matrix Twc
//    points: [N,3], Twc: [4,4], post-multiplied as row vectors.
// -----------------------------------------------------------------------------
void transformPoints(torch::Tensor& points, torch::Tensor& Twc)
{
    namespace idx = torch::indexing;

    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
                "transformPoints: points must be [N,3]");
    TORCH_CHECK(Twc.dim() == 2 && Twc.size(0) == 4 && Twc.size(1) == 4,
                "transformPoints: Twc must be [4,4]");

    auto device = points.device();
    auto N      = points.size(0);

    auto opts = torch::TensorOptions()
                    .dtype(points.dtype())
                    .device(device);

    // Homogeneous coordinates [N,4]
    torch::Tensor ones = torch::ones({N, 1}, opts);
    torch::Tensor pts_h = torch::cat({points, ones}, /*dim=*/1);  // [N,4]

    // Row-vector convention: [N,4] * [4,4] -> [N,4]
    torch::Tensor pts_w = torch::matmul(pts_h, Twc);              // [N,4]

    // Drop homogeneous coordinate
    points = pts_w.index({idx::Slice(), idx::Slice(0, 3)}).contiguous();
}

// -----------------------------------------------------------------------------
// 3) Reproject depth map (pinhole) to 3D points in camera coordinates
//
// depth:            flattened [H*W] depth (meters) on device (CUDA or CPU)
// point_valid_flags:[H*W] bool, currently not used here (kept for signature)
// intr:             fx, fy, cx, cy
// image_width:      W
//
// Return: [H*W,3] tensor of (X,Y,Z) in camera coordinates. You can later
//          mask it with point_valid_flags (as Photo-SLAM does).
// -----------------------------------------------------------------------------
torch::Tensor reprojectDepthPinhole(
    torch::Tensor& depth,
    torch::Tensor& point_valid_flags,
    std::vector<float>& intr,
    int image_width)
{
    namespace idx = torch::indexing;

    TORCH_CHECK(depth.dim() == 1,
                "reprojectDepthPinhole: expected depth to be 1-D flattened [H*W]");
    TORCH_CHECK(intr.size() >= 4,
                "reprojectDepthPinhole: intr must contain at least {fx, fy, cx, cy}");
    TORCH_CHECK(image_width > 0,
                "reprojectDepthPinhole: image_width must be > 0");

    auto device = depth.device();
    auto opts   = torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(device);

    const int64_t N = depth.size(0);
    TORCH_CHECK(N % image_width == 0,
                "reprojectDepthPinhole: depth.size(0) not divisible by image_width");
    const int64_t H = N / image_width;
    const int64_t W = image_width;

    const float fx = intr[0];
    const float fy = intr[1];
    const float cx = intr[2];
    const float cy = intr[3];

    // Build pixel grid
    torch::Tensor u = torch::arange(W, opts)            // [W]
                          .view({1, W})                 // [1,W]
                          .repeat({H, 1});              // [H,W]
    torch::Tensor v = torch::arange(H, opts)            // [H]
                          .view({H, 1})                 // [H,1]
                          .repeat({1, W});              // [H,W]

    u = u.flatten();                                    // [H*W]
    v = v.flatten();                                    // [H*W]

    // Depth
    torch::Tensor z = depth.to(opts);                   // [H*W]

    // Avoid division by zero if intrinsics are weird
    TORCH_CHECK(std::abs(fx) > 1e-8f && std::abs(fy) > 1e-8f,
                "reprojectDepthPinhole: fx/fy must be non-zero");

    torch::Tensor x = (u - cx) / fx * z;                // [H*W]
    torch::Tensor y = (v - cy) / fy * z;                // [H*W]

    // Stack to [H*W,3]
    torch::Tensor points = torch::stack({x, y, z}, /*dim=*/1);  // [H*W,3]

    // point_valid_flags is kept only for signature compatibility
    (void)point_valid_flags;

    return points.contiguous();
}

void VoxelMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<VoxelKeyframe> pkf)
{
    // auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    // Pose of camera in world frame
    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
    {
        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));

        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;

        // Keypoints and local 3D (camera frame)
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(),
            {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(),
            {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f,
            true,
            false);

        // RGB image → torch
        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        // Photo-SLAM’s neighborhood densification
        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor,
                kps_has3D_tensor,
                kps_point_local_tensor,
                colors,
                monocular_inactive_geo_densify_max_pixel_dist_,
                pkf->intr_,
                pkf->image_width_);

        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid   = std::get<1>(result);

        // Transform points to world coordinates
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid},   /*dim=*/0);
        }

        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;

    case STEREO:
    {
        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));

        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // RGB → gray
        cv::cuda::cvtColor(rgb_left_gpu,  gray_left_gpu,  cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // float → uint8
        gray_left_gpu.convertTo(gray_left_gpu,   CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to 3D
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // To torch
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();

        torch::Tensor points3D =
            tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor colors =
            tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        // Keep only points near tracked keypoints + valid disparity range
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)},
            false,
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));

        int nkps_twice = pkf->kps_pixel_.size();
        int width      = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) +
                      static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(
                disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()),
                true,
                false));

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(
                disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()),
                true,
                false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid   = colors.index({point_valid_flags});

        // Transform to world
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid},   /*dim=*/0);
        }

        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;

    case RGBD:
    {
        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));

        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // cv::cuda::GpuMat → torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();

        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        // Filter depth using tracked keypoints + RGBD_min/max
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)},
            false,   // Note Photo-SLAM uses false here and then sets only around kps
            torch::TensorOptions().dtype(torch::kBool).device(device_type_));

        int nkps_twice = pkf->kps_pixel_.size();
        int width      = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(pkf->kps_pixel_[kpidx]) +
                      static_cast<int>(pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }

        // Debug: how many pixels do we mark around kps?
        auto num_kps          = nkps_twice / 2;
        auto num_flags_before = point_valid_flags.sum().item<int64_t>();
        // std::cout << "[RGBD densify] num_kps=" << num_kps
        //         << "  valid_flags_after_kps=" << num_flags_before << std::endl;

        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));
        
        auto num_flags_after_depth = point_valid_flags.sum().item<int64_t>();
        // std::cout << "[RGBD densify] valid_flags_after_depth=" << num_flags_after_depth << std::endl;

        torch::Tensor colors_valid = rgb.index({point_valid_flags});

        // Reproject to 3D (camera coordinates)
        torch::Tensor points3D_valid;
        sv::Camera& camera = scene_->cameras_.at(pkf->camera_id_);

        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_valid = reprojectDepthPinhole(
                depth,
                point_valid_flags,
                pkf->intr_,
                pkf->image_width_);
        }
        break;

        case Camera::FISHEYE:
        {
            // TODO: support fisheye camera?
            throw std::runtime_error("[VoxelMapper] Fisheye cameras are not supported currently!");
        }
        break;

        default:
        {
            throw std::runtime_error("[VoxelMapper] Invalid camera model!");
        }
        break;
        }

        points3D_valid = points3D_valid.index({point_valid_flags});

        // // Debug AFTER we have the points
        // std::cout << "[RGBD densify] points3D_valid shape after reproject+mask: "
        //         << points3D_valid.sizes() << std::endl;
        // std::cout << "[RGBD densify] colors_valid count: "
        //         << colors_valid.size(0) << std::endl;

        // Transform to world coordinates
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Cache
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        } else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid},   /*dim=*/0);
        }

        // savePly(result_dir_ / (std::to_string(getIteration()) + "_" +
        //                       std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;

    default:
    {
        throw std::runtime_error("[VoxelMapper] Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;

        // Add new points to the voxel model
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        voxel_model_->increasePcd(
            depth_cache_points_,
            depth_cache_colors_,
            getIteration());
    }

    // auto end_timing = std::chrono::steady_clock::now();
    // auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     end_timing - start_timing).count();
    // std::cout << "[VoxelMapper] increasePcdByKeyframeInactiveGeoDensify() takes "
    //           << completion_time << " ms" << std::endl;
}

bool VoxelMapper::isStopped() const {
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void VoxelMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

void VoxelMapper::increaseKeyframeTimesOfUse(
        const std::shared_ptr<VoxelKeyframe>& pkf,
        int times)
 {
     pkf->remaining_times_of_use_ += times;
 }

void VoxelMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Voxel Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}

void VoxelMapper::recordKeyframeRendered(
    torch::Tensor&           rendered,
    torch::Tensor&           ground_truth,
    unsigned long            kfid,
    std::filesystem::path    result_img_dir,
    std::filesystem::path    result_gt_dir,
    std::filesystem::path    result_loss_dir,
    std::string              name_suffix)
{
    if (record_rendered_image_) {
         auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
         cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
         image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
     }
 
     if (record_ground_truth_image_) {
         auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
         cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
         gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
     }
 
     if (record_loss_image_) {
         torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
         auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
         cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
         loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
         cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
     }
}

void VoxelMapper::renderAndRecordKeyframe(
    std::shared_ptr<VoxelKeyframe> pkf,
    float&       dssim,
    float&       psnr,
    double&      render_ms,
    const std::filesystem::path& result_img_dir,
    const std::filesystem::path& result_gt_dir,
    const std::filesystem::path& result_loss_dir,
    const std::filesystem::path& result_depth_dir,
    const std::string&           name_suffix)
{
    // std::cout << "pkf image height and width: " << pkf->image_height_ << " " << pkf->image_width_ << std::endl;
    sv::MiniCam cam = pkf->toMiniCam(pkf->image_height_, pkf->image_width_);

    // Render options: ensure depth is requested
    sv::RenderOpts ropts;
    ropts.output_T     = true;   // if you want T for other losses
    ropts.output_depth = true;   // IMPORTANT for depth saving

    auto start_timing = std::chrono::steady_clock::now();
    // Render
    auto render_pkg = voxel_model_->render(
        cam,
        pkf->image_height_,
        pkf->image_width_,
        /* gt_image     */ pkf->original_image_,
        /* color_mode   */ nullptr,
        /* track_max_w  */ false,
        /* ss           */ std::nullopt,
        /* output_depth */ ropts.output_depth,
        /* output_normal*/ false,
        /* output_T     */ ropts.output_T,
        /* rand_bg      */ false,
        /* use_auto_exp */ false,
        ropts
    );
    // auto render_pkg = voxel_model_->render(cam, pkf->image_height_, pkf->image_width_, pkf->original_image_);
    torch::Tensor rendered_image = render_pkg.at("color").to(mDevice);          // (1,3,H,W)
    // Mask and GT on the same device
    torch::Tensor mask = undistort_mask_[pkf->camera_id_]
                            .to(mDevice)
                            .to(torch::kFloat32);                        // (3,H,W) or (1,3,H,W)
    torch::Tensor gt_image = pkf->original_image_.to(mDevice);          // (3,H,W)
    // Broadcast mask over batch if needed
    torch::Tensor masked_image = rendered_image * mask;                 // (1,3,H,W)
    masked_image = masked_image.squeeze(0);                             // (3,H,W)

    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_ms = 1e-6 * render_time_ns;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    

    // ---- Depth saving ----
    torch::Tensor depth_for_viz;
    auto it_depth = render_pkg.find("depth");
    if (it_depth != render_pkg.end() && it_depth->second.defined()) {
        depth_for_viz = it_depth->second;
        // std::cout << "[DEPTH KFs] depth found for kf "
        //             << pkf->fid_ << " shape=" << depth_for_viz.sizes() << "\n";
    } else {
        auto it_raw = render_pkg.find("raw_depth");
        if (it_raw != render_pkg.end() && it_raw->second.defined()) {
            depth_for_viz = it_raw->second;
            // std::cout << "[DEPTH KFs] using raw_depth for kf "
            //             << pkf->fid_ << " shape=" << depth_for_viz.sizes() << "\n";
        } else {
            // std::cout << "[DEPTH KFs] no depth for kf "
            //             << pkf->fid_ << "\n";
        }
    }
    if (depth_for_viz.defined()) {
        std::ostringstream ss;
        ss << "kf_" << std::setw(5) << std::setfill('0') << pkf->fid_ << ".png";
        std::filesystem::path depth_path = result_depth_dir / ss.str();
        saveDepthTensorAsPng(depth_for_viz, depth_path);
    }

    // Per-pixel photometric error (L2 or L1)
    torch::Tensor diff = masked_image - gt_image;  // (3,H,W)
    // Option 1: L2 error per pixel
    torch::Tensor per_pixel_err = diff.pow(2).mean(0);   // [H,W]
    std::filesystem::path heatmap_dir = result_loss_dir.parent_path() / "photo_loss";
    std::filesystem::create_directories(heatmap_dir);
    std::ostringstream err_name;
    err_name << getIteration() << "_"
            << pkf->fid_
            << name_suffix
            << "_photoloss.png";
    std::filesystem::path err_path = heatmap_dir / err_name.str();
    savePhotometricErrorHeatmapAsPng(per_pixel_err, err_path);
 }

void VoxelMapper::renderAndRecordAllKeyframes(const std::string& name_suffix)
{
    // Create result directory with current iteration number and suffix
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir);

    // Create subdirectories if needed
    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);

    // New: depth directory inside the same x_shutdown folder
    std::filesystem::path depth_dir = result_dir / "depth";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(depth_dir);

    // Open logging files
    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Voxel Mapper]Render time statistics: keyframe id, time(milliseconds)\n";

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Voxel Mapper]keyframe id, dssim\n";

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Voxel Mapper]keyframe id, psnr\n";

    // Loop through all keyframes deterministically
    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, render_time, image_dir, image_gt_dir, image_loss_dir, depth_dir, name_suffix);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;

        ++kfit;
    }
}

void VoxelMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    // keyframesToJson(result_dir);
    // saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    // Reconstructed voxel scene
    // voxel_model_->savePly(ply_dir / "point_cloud.ply");
    // Input sparse points (from ORB-SLAM map) for reference
    // voxel_model_->saveSparsePointsPly(result_dir / "input.ply");
}

void VoxelMapper::keyframesToJson(const std::filesystem::path&){ }

/* ---------------- runtime getter / setter ---------------- */
// VariableParameters VoxelMapper::getVariableParameters() const
// {
//     std::lock_guard<std::mutex> lk(mutex_status_);
//     VariableParameters p;
//     p.position_lr_init          = position_lr_init_;
//     p.new_keyframe_times_of_use_ = var_params_.new_keyframe_times_of_use_;
//     p.do_inactive_geo_densify   = do_inactive_geo_densify_;
//     p.keep_training = keep_training_;
//     return p;
// }

// void VoxelMapper::setVariableParameters(const VariableParameters& p)
// {
//     std::lock_guard<std::mutex> lk(mutex_status_);
//     /* apply only what VoxelMapper still honours */
//     position_lr_init_                               = p.position_lr_init;
//     new_keyframe_times_of_use_ = p.new_keyframe_times_of_use_;
//     do_inactive_geo_densify_               = p.do_inactive_geo_densify;
//     keep_training_                         = p.keep_training;
// }

cv::Mat VoxelMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    // Same guard as Photo-SLAM: no rendering before we have something
    if (!initial_mapped_ || getIteration() <= 0) {
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    // Build a temporary keyframe for the viewer pose
    std::shared_ptr<VoxelKeyframe> pkf = std::make_shared<VoxelKeyframe>();
    // pkf->zfar_ = z_far_;   // only if you actually use z_far_ anywhere
    pkf->znear_ = z_near_;

    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());

    try {
        // Camera
        sv::Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // If your VoxelKeyframe has this (like GaussianKeyframe), call it:
        // pkf->computeTransformTensors();
    }
    catch (const std::out_of_range&) {
        throw std::runtime_error("[VoxelMapper::renderFromPose] KeyFrame Camera not found!");
    }

    // Build MiniCam for the viewer resolution
    sv::MiniCam cam = pkf->toMiniCam(height, width);

    // We don't want gradients in the viewer
    torch::NoGradGuard no_grad;

    // Call voxel_model_->render under the same render mutex
    std::unordered_map<std::string, torch::Tensor> pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);

        pkg = voxel_model_->render(
            cam,
            height,
            width,
            /* gt_image      */ torch::Tensor(),  // none
            /* color_mode    */ nullptr,
            /* track_max_w   */ false,
            /* ss            */ std::nullopt,
            /* output_depth  */ false,
            /* output_normal */ false,
            /* output_T      */ false,
            /* rand_bg       */ false,
            /* use_auto_exp  */ false,
            sv::RenderOpts{}   // default options
        );
    }

    // Check we actually got a color image
    auto it = pkg.find("color");
    if (it == pkg.end() || !it->second.defined()) {
        // Fallback: black image
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }

    torch::Tensor color = it->second;  // expected shape [1,3,H,W] or [3,H,W]

    // Masking exactly like GaussianMapper
    torch::Tensor mask;
    if (main_vision) {
        mask = viewer_main_undistort_mask_[pkf->camera_id_];
    } else {
        mask = viewer_sub_undistort_mask_[pkf->camera_id_];
    }

    // Make sure mask is on the same device as color
    if (mask.device() != color.device()) {
        mask = mask.to(color.device());
    }

    // Both should be broadcastable: mask is usually [1,3,H,W] or [3,H,W]
    torch::Tensor masked_image = color * mask;

    // Reuse Photo-SLAM utility to convert to cv::Mat (float32 RGB)
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

// VoxelMapper::~VoxelMapper() {
//     // Explicitly reset any Python or Torch objects that may call Python at destruction
//     voxel_model_.reset();  // Deallocates all tensors and Python wrappers
//     mpSLAM.reset();
// }

int VoxelMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void VoxelMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float VoxelMapper::geoLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.geo_lr_;
}

float VoxelMapper::sh0LearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.sh0_lr_;
}

float VoxelMapper::shsLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.shs_lr_;
}

float VoxelMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}

int VoxelMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.adapt_every_;
}

int VoxelMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}

int VoxelMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}

bool VoxelMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool VoxelMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}

bool VoxelMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

 void VoxelMapper::setgeoLearningRateInit(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.geo_lr_ = lr;
 }
 void VoxelMapper::setsh0LearningRate(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.sh0_lr_ = lr;
 }
 void VoxelMapper::setshsLearningRate(const float lr)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.shs_lr_ = lr;
 }
 void VoxelMapper::setLambdaDssim(const float lambda_dssim)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.lambda_dssim_ = lambda_dssim;
 }

 void VoxelMapper::setDensifyInterval(const int interval)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.adapt_every_ = interval;
 }
 void VoxelMapper::setNewKeyframeTimesOfUse(const int times)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     new_keyframe_times_of_use_ = times;
 }
 void VoxelMapper::setStableNumIterExistence(const int niter)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     stable_num_iter_existence_ = niter;
 }
 void VoxelMapper::setKeepTraining(const bool keep)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     keep_training_ = keep;
 }
 void VoxelMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     do_gaus_pyramid_training_ = gaus_pyramid;
 }
 
 VariableParameters VoxelMapper::getVaribleParameters()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     VariableParameters params;
     params.geo_lr = opt_params_.geo_lr_;
     params.sh0_lr = opt_params_.sh0_lr_;
     params.shs_lr = opt_params_.shs_lr_;
     params.lambda_dssim = opt_params_.lambda_dssim_;
     params.densify_interval = opt_params_.adapt_every_;
     params.new_kf_times_of_use = new_keyframe_times_of_use_;
     params.stable_num_iter_existence = stable_num_iter_existence_;
     params.keep_training = keep_training_;
     params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
     return params;
 }
 
 void VoxelMapper::setVaribleParameters(const VariableParameters &params)
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     opt_params_.geo_lr_ = params.geo_lr;
     opt_params_.sh0_lr_ = params.sh0_lr;
     opt_params_.shs_lr_ = params.shs_lr;
     opt_params_.lambda_dssim_ = params.lambda_dssim;
     opt_params_.adapt_every_ = params.densify_interval;
     new_keyframe_times_of_use_ = params.new_kf_times_of_use;
     stable_num_iter_existence_ = params.stable_num_iter_existence;
     keep_training_ = params.keep_training;
     do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
 }

// void VoxelMapper::saveVoxelErrorHeatmap(const sv::MiniCam& cam,
//                                         const torch::Tensor& rendered_img, // (1,3,H,W) float[0..1], device=mDevice
//                                         const torch::Tensor& gt_img,       // (1,3,H,W) float[0..1], device=mDevice
//                                         int fid,
//                                         const std::string& base_dir)
// {
//     namespace fs = std::filesystem;
//     const fs::path dir_kf = fs::path(base_dir) / ("kf" + std::to_string(fid));
//     fs::create_directories(dir_kf);
//     torch::NoGradGuard no_grad;

//     const int H = gt_img.size(2);
//     const int W = gt_img.size(3);

//     // 1) per-pixel MSE (current frame)
//     torch::Tensor mse_map = (rendered_img - gt_img).pow(2).mean(1); // (1,H,W) on device
//     mse_map = mse_map.squeeze(0).detach().to(torch::kCPU).contiguous(); // (H,W) CPU

//     // 2) Build approximate per-pixel -> voxel index map
//     auto geom_cpu = approxGeomFromCentersAndSize(
//     cam,
//     voxel_model_->voxCenter(),         // [N,3]
//     voxel_model_->voxSize(),           // [N,1]
//     H, W
// );

//     // 3) Reduce per-pixel error -> per-voxel error
//     const int64_t N = voxel_model_->numVoxels();
//     auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

//     torch::Tensor idx_flat  = geom_cpu.view({-1});                       // (H*W) int64
//     torch::Tensor valid     = idx_flat >= 0;
//     torch::Tensor sel_idx   = idx_flat.index({valid});                   // (K)
//     torch::Tensor err_flat  = mse_map.view({-1}).index({valid});         // (K)

//     torch::Tensor vox_sum   = torch::zeros({N}, opts_f32);               // per-voxel sum
//     torch::Tensor vox_count = torch::zeros({N}, opts_f32);               // per-voxel hit count
//     if (sel_idx.numel() > 0) {
//         vox_sum.index_add_(0, sel_idx, err_flat);
//         vox_count.index_add_(0, sel_idx, torch::ones_like(err_flat));
//     }
//     torch::Tensor vox_err = vox_sum / vox_count.clamp_min(1.0f);         // (N)

//     // 4) Map per-voxel error back to pixels for visualization
//     torch::Tensor safe_idx = idx_flat.clone();
//     safe_idx.masked_fill_(~valid, 0);
//     torch::Tensor pix_err  = vox_err.index_select(0, safe_idx).view({H, W});
//     pix_err.masked_fill_(~valid.view({H, W}), 0);

//     float vmin = pix_err.min().item<float>();
//     float vmax = pix_err.max().item<float>();
//     float range = std::max(1e-8f, vmax - vmin);
//     torch::Tensor H01 = (pix_err - vmin) / range;

//     // 5) Jet map (B,G,R) like before
//     auto R = (1.5f * H01 - 0.5f).clamp(0, 1);
//     auto G = (1.5f - (2 * H01 - 1).abs()).clamp(0, 1);
//     auto B = (0.5f - 1.5f * H01).clamp(0, 1);
//     torch::Tensor rgb = torch::stack({B, G, R}, -1) * 255.0f; // (H,W,3) BGR for OpenCV
//     rgb = rgb.to(torch::kUInt8).cpu().contiguous();

//     cv::Mat img(H, W, CV_8UC3, rgb.data_ptr<uint8_t>());

//     // 6) Legend bar
//     const int LWIDTH = 32;
//     cv::Mat legend(H, LWIDTH, CV_8UC3);
//     for (int y = 0; y < H; ++y) {
//         float val = 1.f - float(y) / float(H - 1);
//         float r = std::clamp( 1.5f*val - 0.5f , 0.f, 1.f),
//               g = std::clamp( 1.5f - std::abs(2*val -1) , 0.f, 1.f),
//               b = std::clamp( 0.5f - 1.5f*val , 0.f, 1.f);
//         legend.row(y).setTo(cv::Vec3b{ uint8_t(255*b), uint8_t(255*g), uint8_t(255*r) });
//     }
//     cv::putText(legend, "high loss (red)", {2, 14},        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1, cv::LINE_AA);
//     cv::putText(legend, "low loss (blue)", {2, H - 6},     cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1, cv::LINE_AA);

//     cv::Mat out; cv::hconcat(img, legend, out);

//     const int x0 = W + LWIDTH + 4;
//     auto put = [&](float frac, const std::string& txt)
//     {
//         int y = int((1.f - frac) * (H - 1));
//         cv::putText(out, txt, {x0, y}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1, cv::LINE_AA);
//         cv::line(out, {W, y}, {W + LWIDTH - 1, y}, cv::Scalar(255,255,255), 1, cv::LINE_AA);
//     };

//     std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(3);
//     ss << vmax;                put(1.f,   ss.str()); ss.str(""); ss.clear();
//     ss << vmax - 0.25f*range;  put(0.75f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin + 0.50f*range;  put(0.50f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin + 0.25f*range;  put(0.25f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin;                put(0.f,   ss.str());

//     // 7) Save PNGs
//     std::ostringstream fn;
//     fn << "kf"   << std::setw(4) << std::setfill('0') << fid
//        << "_iter"<< std::setw(6) << std::setfill('0') << iteration_
//        << ".png";
//     cv::imwrite((dir_kf / fn.str()).string(), out);

//     // Save GT image for this KF
//     auto gt = (gt_img.squeeze(0).mul(255.0f).clamp(0.0f,255.0f).to(torch::kUInt8).permute({1,2,0}).cpu().contiguous());
//     cv::Mat gt_mat(H, W, CV_8UC3, gt.data_ptr<uint8_t>());
//     cv::cvtColor(gt_mat, gt_mat, cv::COLOR_RGB2BGR);
//     std::ostringstream fn2;
//     fn2<< "kf"<<std::setw(4)<<std::setfill('0')<<fid
//        << "_iter"<<std::setw(6)<<std::setfill('0')<<iteration_
//        << "_gt.png";
//     cv::imwrite((dir_kf / fn2.str()).string(), gt_mat);
// }

// void VoxelMapper::saveVoxelErrorHeatmap(const sv::MiniCam&  /*cam*/,
//                                         const torch::Tensor& geom,
//                                         const torch::Tensor&  gt_img,
//                                         int                 fid,          // NEW
//                                         const std::string&  base_dir)     // ← e.g. result_dir_/heatmaps
// {
//     namespace fs = std::filesystem;
//     const fs::path dir_kf = fs::path(base_dir) / ("kf" + std::to_string(fid));
//     fs::create_directories(dir_kf);       // <-- makes .../heatmaps/kf<i>
//     torch::NoGradGuard no_grad;

//     /* ------------------------------------------------------------------ *
//      * ❶  error per voxel  →  per-pixel array  (H,W) in [vmin,vmax]        *
//      * ------------------------------------------------------------------ */
//     torch::Tensor vox_err = (voxel_model_->voxel_error_sum_
//                             / voxel_model_->voxel_hit_count_.clamp_min(1))
//                                 .squeeze(1);                                 // (N)

//     const int H = geom.size(0), W = geom.size(1);

//     torch::Tensor idx_flat = geom.view(-1).to(torch::kLong);
//     torch::Tensor valid    = idx_flat >= 0;
//     torch::Tensor safe_idx = idx_flat.clone().masked_fill(~valid, 0);

//     torch::Tensor pix_err  = vox_err.index_select(0, safe_idx)
//                                    .view({H, W});
//     pix_err.masked_fill_(~valid.view({H, W}), 0);

//     float vmin = pix_err.min().item<float>(),
//           vmax = pix_err.max().item<float>(),
//           range= std::max(1e-6f, vmax - vmin);

//     torch::Tensor H01 = (pix_err - vmin) / range;            // → [0,1]

//     /* ------------------------------------------------------------------ *
//      * ❷  Jet colour-map  (same formula as before)                         *
//      * ------------------------------------------------------------------ */
//     auto R = (1.5f * H01 - 0.5f).clamp(0, 1);
//     auto G = (1.5f - (2 * H01 - 1).abs()).clamp(0, 1);
//     auto B = (0.5f - 1.5f * H01).clamp(0, 1);
//     torch::Tensor rgb = torch::stack({B, G, R}, -1) * 255.0f;  // BGR for OpenCV
//     rgb = rgb.to(torch::kUInt8).cpu().contiguous();            // (H,W,3)

//     /* ------------------------------------------------------------------ *
//      * ❸  convert to cv::Mat                                              *
//      * ------------------------------------------------------------------ */
//     cv::Mat img(H, W, CV_8UC3, rgb.data_ptr<uint8_t>());

//     /* ------------------------------------------------------------------ *
//      * ❹  legend bar  (32 px wide)                                         *
//      * ------------------------------------------------------------------ */
//     const int LWIDTH = 32;
//     cv::Mat legend(H, LWIDTH, CV_8UC3);

//     for (int y = 0; y < H; ++y)
//     {
//         float val = 1.f - float(y) / float(H - 1);   // top=max (red), bottom=min (blue)
//         float r = std::clamp( 1.5f*val - 0.5f , 0.f, 1.f),
//               g = std::clamp( 1.5f - std::abs(2*val -1) , 0.f, 1.f),
//               b = std::clamp( 0.5f - 1.5f*val , 0.f, 1.f);
//         cv::Vec3b col{ uint8_t(255*b), uint8_t(255*g), uint8_t(255*r) };
//         legend.row(y).setTo(col);
//     }

//     const std::string lbl_hi = "high loss (red)";
//     const std::string lbl_lo = "low loss (blue)";
//     // near the top of the legend bar:
//     cv::putText(legend, lbl_hi, {2, 14},
//                 cv::FONT_HERSHEY_SIMPLEX, 0.45,
//                 cv::Scalar(255,255,255), 1, cv::LINE_AA);
//     // near the bottom of the legend bar:
//     cv::putText(legend, lbl_lo, {2, H - 6},
//                 cv::FONT_HERSHEY_SIMPLEX, 0.45,
//                 cv::Scalar(255,255,255), 1, cv::LINE_AA);

//     /* ------------------------------------------------------------------ *
//      * ❺  stack data + legend & annotate tick labels                       *
//      * ------------------------------------------------------------------ */
//     cv::Mat out;
//     cv::hconcat(img, legend, out);

//     const int x0 = W + LWIDTH + 4;        // text anchor (pixels from left)
//     auto put = [&](float frac, const std::string& txt)
//     {
//         int y = int((1.f - frac) * (H - 1));
//         cv::putText(out, txt, {x0, y},
//                     cv::FONT_HERSHEY_SIMPLEX, 0.45,
//                     cv::Scalar(255,255,255), 1, cv::LINE_AA);
//         cv::line(out,
//                  {W, y}, {W + LWIDTH - 1, y},
//                  cv::Scalar(255,255,255), 1, cv::LINE_AA);
//     };

//     std::ostringstream ss; ss.setf(std::ios::fixed); ss.precision(3);
//     ss << vmax; put(1.f, ss.str());  ss.str(""); ss.clear();
//     ss << vmax - 0.25f*range; put(0.75f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin + 0.50f*range; put(0.50f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin + 0.25f*range; put(0.25f, ss.str()); ss.str(""); ss.clear();
//     ss << vmin;  put(0.f, ss.str());

//     /* ------------------------------------------------------------------ *
//      * ❻  write png                                                       *
//      * ------------------------------------------------------------------ */
//     std::ostringstream fn;
//     fn << "kf"   << std::setw(4) << std::setfill('0') << fid
//        << "_iter"<< std::setw(6) << std::setfill('0') << iteration_
//        << ".png";

//     cv::imwrite((dir_kf / fn.str()).string(), out);

//         // ————————————— ❼ convert & write GT image —————————————
//     // assume gt_img is (1,3,H,W) float in [0,1]
//     auto gt = (gt_img.squeeze(0).mul(255.0f)
//                    .clamp(0.0f,255.0f)
//                    .to(torch::kUInt8)
//                    .permute({1,2,0})            // H,W,3 RGB
//                    .cpu()
//                    .contiguous());
//     // convert to BGR for OpenCV:
//     cv::Mat gt_mat(H, W, CV_8UC3, gt.data_ptr<uint8_t>());
//     cv::cvtColor(gt_mat, gt_mat, cv::COLOR_RGB2BGR);

//     std::ostringstream fn2;
//     fn2<< "kf"<<std::setw(4)<<std::setfill('0')<<fid
//        << "_iter"<<std::setw(6)<<std::setfill('0')<<iteration_
//        << "_gt.png";
//     cv::imwrite((dir_kf / fn2.str()).string(), gt_mat);
// }