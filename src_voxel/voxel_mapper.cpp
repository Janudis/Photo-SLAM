#include "include_voxel/voxel_mapper.h"
#include <limits>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <regex>

namespace py = pybind11;
std::ofstream loss_log_;
std::ofstream loss_l1_log_;
std::ofstream loss_ssim_log_;
std::ofstream loss_l2_log_;

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

template<typename KFMap, typename CamContainer>
void dumpKeyframesForProjectionFile(const KFMap& kfmap,
                                    const CamContainer& cameras,
                                    const std::filesystem::path& out_dir)
{
    std::lock_guard<std::mutex> lk(g_dumpkf_mutex);
    std::filesystem::create_directories(out_dir);
    std::filesystem::create_directories(out_dir / "imgs");
    const auto tmp_file = out_dir / "keyframes_proj.tmp";
    const auto out_file = out_dir / "keyframes_proj.txt";
    std::ofstream os(tmp_file, std::ios::trunc);
    if (!os) { std::cerr << "[dumpKF] cannot open " << tmp_file << '\n'; return; }
    size_t n_lines = 0;
    for (const auto& kv : kfmap) {
        const auto& kf_ptr = kv.second;
        if (!kf_ptr) continue;
        const auto cam_id = kf_ptr->camera_id_;
        auto cam_it = cameras.find(cam_id);
        if (cam_it == cameras.end()) continue;
        const sv::Camera& cam = cam_it->second;
        const int   W  = kf_ptr->image_width_;
        const int   H  = kf_ptr->image_height_;
        const float fx = cam.fx(), fy = cam.fy(), cx = cam.cx(), cy = cam.cy();
        const Eigen::Matrix4f Tcw = kf_ptr->getWorld2View2(kf_ptr->trans_, kf_ptr->scale_);
        std::ostringstream oss;
        oss << "imgs/kf_" << kf_ptr->fid_ << ".png";
        const std::string rel_img = oss.str();
        os << kf_ptr->fid_ << ' '
           << W << ' ' << H << ' '
           << std::setprecision(9) << fx << ' ' << fy << ' ' << cx << ' ' << cy << ' ';
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                os << Tcw(r,c) << ' ';
        os << rel_img << '\n';
        ++n_lines;
    }
    os.close();
    std::error_code ec;
    std::filesystem::rename(tmp_file, out_file, ec);
    if (ec) {
        std::filesystem::copy_file(tmp_file, out_file,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tmp_file);
    }
    // std::cout << "[dumpKF] wrote " << out_file << " (" << n_lines << " lines)\n";
}
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
    std::filesystem::create_directories(extrema_dir_);

    switch (pSLAM->getSensorType()) {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
        sensor_type_ = MONOCULAR;
        break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
        sensor_type_ = STEREO;
        break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
        sensor_type_ = RGBD;
        break;
    default:
        throw std::runtime_error("[Voxel Mapper]Unsupported sensor type!");
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

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new);

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

    pipe_params_.convert_SHs_ =
         (settings_file["Pipeline.convert_SHs"].operator int()) != 0;

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

void VoxelMapper::run()
{
    /* expose our helper scripts to the embedded Python side */
    py::gil_scoped_acquire gil;
    py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");

    /* ───────────────────────────────────────────────
     *  1.  INITIAL VOXEL   M A P P I N G  LOOP
     * ─────────────────────────────────────────────── */
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

            // std::vector<std::shared_ptr<VoxelKeyframe>> keyframes_for_bounding;
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
                    cv::Mat imgRGB_undistorted;
                    // Debug: Print Keyframe pose
                    // std::cout << "[DEBUG] Keyframe ID: " << pKF->mnId 
                    //           << " Pose: (" << pose.translation().x() << ", " << pose.translation().y() << ", " << pose.translation().z() << ")\n";
                    // Camera
                    sv::Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                    new_kf->setCameraParams(camera);
                    // Image (left if STEREO)
                    cv::Mat imgRGB = pKF->imgLeftRGB;
                    camera.undistortImage(imgRGB, imgRGB_undistorted);

                    {
                        static const auto proj_dir = result_dir_ / "proj_debug";
                        static const auto imgs_dir = proj_dir / "imgs";
                        std::filesystem::create_directories(imgs_dir);
                        // imgRGB_undistorted is CV_32FC3 RGB in [0..1] (in Photo-SLAM). If it’s 8U, convert first:
                        cv::Mat img_float;
                        if (imgRGB_undistorted.type() == CV_32FC3) {
                            img_float = imgRGB_undistorted;
                        } else {
                            imgRGB_undistorted.convertTo(img_float, CV_32FC3, 1.0/255.0);
                        }
                        saveKfPng_fromFloatRGB(img_float, pKF->mnId, imgs_dir);
                    }

                    new_kf->original_image_ =
                        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                    new_kf->img_filename_ = pKF->mNameFile;

                    // Compute transformations
                    // new_kf->computeTransformTensors();
                    scene_->addKeyframe(new_kf, &kfid_shuffled_);

                    increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());
                }
            }   // Mutex released

                aabb_min_.setConstant( std::numeric_limits<float>::infinity());
                aabb_max_.setConstant(-std::numeric_limits<float>::infinity());
                have_bounds_ = false;
                for (const auto& kv : scene_->cached_point_cloud_) {
                    // cast<float>() returns a temporary → store as a VALUE, not a const&
                    Eigen::Vector3f P = kv.second.xyz_.cast<float>();
                    extendAABB(aabb_min_, aabb_max_, P);
                    have_bounds_ = true;
                }
                if (have_bounds_) {
                    std::cout.setf(std::ios::fixed);
                    // std::cout << std::setprecision(6)
                    //         << "[AABB:init] min:[" << aabb_min_.x() << "," << aabb_min_.y() << "," << aabb_min_.z()
                    //         << "] max:[" << aabb_max_.x() << "," << aabb_max_.y() << "," << aabb_max_.z() << "]\n";
                }
                dumpKeyframesForProjectionFile(
                    scene_->keyframes(),        // all KFs currently in the scene
                    scene_->cameras_,           // intrinsics you already use
                    result_dir_ / "proj_debug"  // output folder
                );

                std::vector<sv::MiniCam> tr_cams;
                tr_cams.reserve(scene_->keyframes().size());
                for (auto& kv : scene_->keyframes()) {
                    if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
                }

                // D) Create voxel model & trainer setup
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                    std::cout << "[VoxelMapper] Scene extent: " 
                              << scene_->cameras_extent_ << std::endl;
                    // save_initial_pcd_npy(result_dir_, scene_->cached_point_cloud_);
                    auto restored = load_full_pcd_from_logs((result_dir_ / "offline_experiment").string());
                    // voxel_model_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_, keyframes_for_bounding, (result_dir_ / "training_camera_poses.txt").string());
                    voxel_model_->createFromPcd(scene_->cached_point_cloud_);
                    // voxel_model_->createFromPcd(std::move(restored));
                    std::unique_lock<std::mutex> lock(mutex_settings_);
                    voxel_model_->createTrainer(
                                                    opt_params_.geo_lr_,
                                                    opt_params_.sh0_lr_,
                                                    opt_params_.shs_lr_,
                                                    opt_params_.optim_beta1_,
                                                    opt_params_.optim_beta2_,
                                                    opt_params_.optim_eps_,
                                                    opt_params_.lr_decay_ckpt_,     // milestones (vector<int>)
                                                    opt_params_.lr_decay_mult_      // gamma
                                                );
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
    /* ───────────────────────────────────────────────
     *  2.  INCREMENTAL   M A P P I N G  LOOP
     * ───────────────────────────────────────────── */
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
    /* ───────────────────────────────────────────────
     *  3.  TAIL   O P T I M I S A T I O N
     * ───────────────────────────────────────────── */
    int adapt_interval = std::max(1, opt_params_.adapt_every_);          // cfg.procedure.adapt_every
    int n_delay_iters  = static_cast<int>(adapt_interval * 0.8f);        // same heuristic as GS code

    while (  (getIteration() - SLAM_stop_iter) <= n_delay_iters
        || (getIteration() % adapt_interval) <= n_delay_iters
        || isKeepingTraining() )
    {
        trainForOneIteration();
        // Re-read in case user changed cfg at runtime
        adapt_interval = std::max(1, opt_params_.adapt_every_);
        n_delay_iters  = static_cast<int>(adapt_interval * 0.8f);
    }

    if (have_bounds_) {
        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(6)
                << "[AABB:final] min:[" << aabb_min_.x() << "," << aabb_min_.y() << "," << aabb_min_.z()
                << "] max:[" << aabb_max_.x() << "," << aabb_max_.y() << "," << aabb_max_.z() << "]\n";
    }

     // Save and clear
     renderAndRecordAllKeyframes("_shutdown");
     savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
     writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
    //  pose_dump_stream_.close();
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

// Add training with optimization loop
void VoxelMapper::trainForOneIteration()
{
    // 1) bump global iteration counter
    increaseIteration(1);

    // 2) pick a random keyframe from the sliding window
    std::shared_ptr<VoxelKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        // if none available, roll back iteration and exit
        increaseIteration(-1);
        return;
    }
    writeKeyframeUsedTimes(result_dir_ / "used_times");
    const int iter = getIteration();
    // bool saved_this_iter = false;
    // 3) select ground truth image + mask tensors
    //    (it always use the “original” resolution in our voxel case)
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    image_height = viewpoint_cam->image_height_;
    image_width = viewpoint_cam->image_width_;
    gt_image = viewpoint_cam->original_image_
                                .to(mDevice)          // (3,H,W)
                                .unsqueeze(0);        // → (1,3,H,W)
    mask = undistort_mask_[viewpoint_cam->camera_id_]
                                .to(mDevice)
                                .to(torch::kFloat32); // (H,W)
    // if it somehow came in as 3×H×W, just take the first (they're identical)
    if (mask.dim() == 3 && mask.size(0) == 3) {
        mask = mask[0];   // now (H,W)
    }
    // now make it 1×1×H×W
    if (mask.dim() == 2) {
        mask = mask.unsqueeze(0).unsqueeze(0);
    }
    else if (mask.dim() == 3) {
        // if somebody gave you (1,H,W) already:
        mask = mask.unsqueeze(1);  // (1,1,H,W)
    }

    // 4) grow SH degree every 1000 iterations (locked during render)
    std::unique_lock<std::mutex> lock_render(mutex_render_);
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
    {
        default_sh_ += 1;
        std::cout << "[VoxelMapper] SH degree: " << default_sh_ << std::endl;
    }    
    voxel_model_->setShDegree(default_sh_);

    // ) build a MiniCam out of this keyframe
    sv::MiniCam cam = viewpoint_cam->toMiniCam();
    // py::module_ np = py::module_::import("numpy");
    // torch::Tensor chw_u8 = viewpoint_cam->original_image_    // (3,H,W) in [0,1]
    //                         .mul(255.0f)
    //                         .clamp(0.0f, 255.0f)
    //                         .to(torch::kUInt8)            // <--- cast to U8
    //                         .cpu()
    //                         .contiguous();
    // torch::Tensor hwc_u8 = chw_u8.permute({1,2,0}).contiguous();  // (H,W,3)
    // py::array rgb_numpy = sv::tensorToNumpyRGB(hwc_u8);
    // np.attr("save")("/home/dimitris/Photo-SLAM/gt_image.npy", rgb_numpy);

    auto render_pkg = voxel_model_->render(cam);
    if (render_pkg.empty() || !render_pkg.count("color") || !render_pkg.at("color").defined()) {
        std::cout << "render pkg empty" << std::endl;
        return;
    }
    // keep running max_w stats for pruning diagnostics (if returned)
    if (render_pkg.count("max_w") && render_pkg.at("max_w").defined()) {
        voxel_model_->max_w_ = torch::maximum(
            voxel_model_->max_w_, render_pkg["max_w"].to(mDevice));
    }

    torch::Tensor rendered_image = render_pkg["color"].to(mDevice);
    torch::Tensor masked_image = rendered_image * mask;      // (1,3,H,W)
    torch::Tensor masked_gt   = gt_image * mask;
    
    // saveVoxelErrorHeatmap(cam, masked_image.detach(), masked_gt.detach(),
    //                     viewpoint_cam->fid_, (result_dir_ / "heatmaps").string());
    // saved_this_iter = true;

    // --- Save render for the FIRST keyframe ---
    // {
    //     torch::NoGradGuard no_grad;
    //     namespace fs = std::filesystem;

    //     // Resolve the first keyframe id once (min id in the map)
    //     static int first_fid = -1;
    //     if (first_fid < 0) {
    //         const auto& kfs = scene_->keyframes();
    //         if (!kfs.empty()) first_fid = kfs.begin()->first; // std::map => smallest id
    //     }

    //     // Save only if this iteration actually trained on the first KF
    //     if (first_fid >= 0 && viewpoint_cam->fid_ == first_fid) {
    //         const int iter = getIteration();

    //         // Folder name is robust to nonzero first_fid
    //         std::ostringstream kf_dir_name;
    //         kf_dir_name << "kf" << std::setw(4) << std::setfill('0') << first_fid;
    //         fs::path dir_kf = result_dir_ / "renders" / kf_dir_name.str();
    //         fs::create_directories(dir_kf);

    //         // Filename with iter stamp
    //         std::ostringstream fn;
    //         fn << "kf" << std::setw(4) << std::setfill('0') << first_fid
    //         << "_iter" << std::setw(6) << std::setfill('0') << iter << ".png";

    //         std::cout << "[KF-LOG] iter " << std::setw(6) << std::setfill('0') << getIteration()
    //           << " trained on KF " << viewpoint_cam->fid_
    //           << " (first_fid=" << first_fid << ")\n";

    //         // Use the render you already computed for this KF
    //         saveDebugImage(render_pkg.at("color"), (dir_kf / fn.str()).string());

    //         // Save GT once for this KF
    //         static bool gt_saved_once = false;
    //         if (!gt_saved_once) {
    //             saveDebugImage(viewpoint_cam->original_image_.unsqueeze(0),
    //                         (dir_kf / "gt.png").string());
    //             gt_saved_once = true;
    //         }
    //     }
    // }

    auto Ll1 = loss_utils::l1_loss(rendered_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto photoslam_loss = (1.0 - lambda_dssim) * Ll1
            + lambda_dssim * (1.0 - loss_utils::ssim(rendered_image, gt_image, mDevice.type()));
    auto mse  = loss_utils::l2_loss(masked_image, masked_gt);
    auto loss = mse; 

    if (!rendered_image.requires_grad()) {
        std::cerr << "[warn] rendered_image.requires_grad == false; grad_fn="
                << (rendered_image.grad_fn() ? "set" : "NULL") << "\n";
    }

    voxel_model_->optimizerZeroGrad();   // move this BEFORE backward
    {
        py::gil_scoped_release no_gil;
        loss.backward();
    }

    // Sanity probe: are grads on the actual Python params?
    // Optional: pre-step grad check
    // if (iter % 10 == 1) {
    //     std::cout << "[dbg] pre-step: ||_sh0.grad||=" << voxel_model_->gradL2("_sh0")
    //             << "  ||_shs.grad||=" << voxel_model_->gradL2("_shs")
    //             << "  ||_geo.grad||=" << voxel_model_->gradL2("_geo_grid_pts") << "\n";
    // }
    // /* ===================== step-delta probe (place here) ===================== */
    // static torch::Tensor sh0_prev, shs_prev, geo_prev;
    // if (iter % 100 == 1) {
    //     sh0_prev = voxel_model_->snapParam("_sh0");
    //     shs_prev = voxel_model_->snapParam("_shs");
    //     geo_prev = voxel_model_->snapParam("_geo_grid_pts");
    // }

    if (opt_params_.lambda_tv_density_ > 0.f &&
        iter >= opt_params_.tv_from_ &&
        iter <= opt_params_.tv_until_) {
        voxel_model_->applyTvOnDensityField(opt_params_.lambda_tv_density_);
    }
    // if (opt_params_.lambda_tv_density_ > 0.f &&
    //     iter >= opt_params_.tv_from_ &&
    //     iter <= opt_params_.tv_until_) {
    //     if ((iter % 200) == 0) {
    //         std::cout << "[tv] ||_geo.grad|| before TV = "
    //                 << voxel_model_->gradL2("_geo_grid_pts") << "\n";
    //         voxel_model_->applyTvOnDensityField(opt_params_.lambda_tv_density_);
    //         std::cout << "[tv] ||_geo.grad||  after TV = "
    //                 << voxel_model_->gradL2("_geo_grid_pts") << "\n";
    //     } else {
    //         voxel_model_->applyTvOnDensityField(opt_params_.lambda_tv_density_);
    //     }
    // }

    voxel_model_->optimizerStep();   // <-- the actual update happens here
    // std::cout << "geo_lr: " << geo_lr << " sh0_lr: " << sh0_lr << " shs_lr: " << shs_lr << std::endl;

    // if (iter % 100 == 1) {
    //     std::cout << std::fixed << std::setprecision(6)
    //             << "[step∆] rel||sh0||=" << voxel_model_->deltaFrom("_sh0", sh0_prev)
    //             << "  rel||shs||="      << voxel_model_->deltaFrom("_shs", shs_prev)
    //             << "  rel||geo||="      << voxel_model_->deltaFrom("_geo_grid_pts", geo_prev)
    //             << "\n";
    // }
    // // optional deep debug
    // if (iter % 50 == 1)  voxel_model_->debugParamChain();
    // if (iter % 200 == 1) voxel_model_->debugOptimizer();

    {
        // Densification for increasePcd
        const bool meet_adapt_period =
            (iter % opt_params_.adapt_every_ == 0) &&
            (iter >= opt_params_.adapt_from_)     &&
            (iter <= opt_params_.iterations_ - 500);

        const bool need_pruning =
            meet_adapt_period && (iter <= opt_params_.prune_until_);

        const bool need_subdividing =
            meet_adapt_period &&
            (iter <= opt_params_.subdivide_until_) &&
            (voxel_model_->numVoxels() < opt_params_.subdivide_max_num_);

        py::object sched_state;  // keep in outer scope of the densification block
        if (need_pruning || need_subdividing)
        {
            // Build list of training cameras (use all current keyframes)
            std::vector<sv::MiniCam> tr_cams; tr_cams.reserve(scene_->keyframes().size());
            for (auto& kv : scene_->keyframes()) {
                if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
            }

            // Compute statistics once (max_w, min_samp_interval, view_cnt)
            auto stat = voxel_model_->computeTrainingStat(tr_cams);
            sched_state = voxel_model_->schedulerStateDict();

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

                const int ori_n = voxel_model_->numVoxels();
                auto prune_mask = (stat.max_w < prune_thres).squeeze(1); // [N] bool

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
        }
    }

    // Update learning rate
    voxel_model_->schedulerStep();
    auto [geo_lr, sh0_lr, shs_lr] = voxel_model_->currentLearningRates();
    // std::cout << "[lr] geo=" << geo_lr << " sh0=" << sh0_lr << " shs=" << shs_lr << "\n";
    if (mDevice == torch::kCUDA) torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item<float>() + 0.6f * ema_loss_for_log_;
        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image,
                                    gt_image,
                                    viewpoint_cam->fid_,
                                    result_dir_, result_dir_, result_dir_);
        // every training_report_interval_ iterations, print a concise report
        // if (training_report_interval_ && iteration_ % training_report_interval_ == 0) {
        //     sv::VoxelTrainer::trainingReport(
        //         iteration_,
        //         opt_params_.iterations_,
        //         Ll1,
        //         loss,
        //         ema_loss_for_log_,
        //         loss_utils::l1_loss,
        //         // elapsed_ms,
        //         *voxel_model_,
        //         *scene_,
        //         pipe_params_,
        //         background_
        //     );
        // }
        
        // Extract scalars
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

        if (training_report_interval_ && iteration_ % training_report_interval_ == 0) {
            std::cout << "[TRAIN] iter " << iteration_
                    << "  L1: "    << Ll1.item<float>()
                    << "  MSE: "   << mse.item<float>()
                    << "  loss: "  << loss.item<float>()
                    << "  ema: "   << ema_loss_for_log_ << '\n';
        }

        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(iteration_) / "ply");
        }
        
        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;
    }
}

    // {
    //     // Densification
    //     const bool meet_adapt_period =
    //         (iter % opt_params_.adapt_every_ == 0) &&
    //         (iter >= opt_params_.adapt_from_)     &&
    //         (iter <= opt_params_.iterations_ - 500);

    //     const bool need_pruning =
    //         meet_adapt_period && (iter <= opt_params_.prune_until_);

    //     const bool need_subdividing =
    //         meet_adapt_period &&
    //         (iter <= opt_params_.subdivide_until_) &&
    //         (voxel_model_->numVoxels() < opt_params_.subdivide_max_num_);

    //     py::object sched_state;  // keep in outer scope of the densification block
    //     if (need_pruning || need_subdividing)
    //     {
    //         bool changed = false;
    //         // Build list of training cameras (use all current keyframes)
    //         std::vector<sv::MiniCam> tr_cams; tr_cams.reserve(scene_->keyframes().size());
    //         for (auto& kv : scene_->keyframes()) {
    //             if (kv.second) tr_cams.push_back(kv.second->toMiniCam());
    //         }

    //         // Compute statistics once (max_w, min_samp_interval, view_cnt)
    //         auto stat = voxel_model_->computeTrainingStat(tr_cams);
    //         sched_state = voxel_model_->schedulerStateDict();

    //         std::cout << "[densify:stat] N=" << voxel_model_->numVoxels()
    //         << "  max_w=" << shp(stat.max_w)
    //         << "  min_itv=" << shp(stat.min_samp_interval)
    //         << "  view_cnt=" << shp(stat.view_cnt) << "\n";

    //         // PRUNE
    //         torch::Tensor prune_mask; // keep for subdiv mask logic
    //         if (need_pruning) {
    //             const int   a0 = opt_params_.adapt_from_;
    //             const int   a1 = opt_params_.prune_until_;
    //             const float t0 = opt_params_.prune_thres_init_;
    //             const float t1 = opt_params_.prune_thres_final_;

    //             const float prune_thres = (iter <= a0) ? t0 :
    //                                     (iter >= a1) ? t1 :
    //                                     (t0 + (t1 - t0) * float(iter - a0) / float(std::max(1, a1 - a0)));

    //             int ori_n = voxel_model_->numVoxels();
    //             prune_mask = (stat.max_w < prune_thres).squeeze(1); // [N] bool

    //             int n_can = stat.max_w.size(0);
    //             int n_prune = (int)(prune_mask.defined()? prune_mask.sum().item<int64_t>() : -1);
    //             std::cout << "[PRUNE:prep] thresh=" << prune_thres
    //                     << "  N=" << n_can
    //                     << "  prune_sum=" << n_prune << "\n";

    //             voxel_model_->pruning(prune_mask);
    //             int new_n = voxel_model_->numVoxels();

    //             std::cout << "[PRUNING]     " << std::setw(7) << ori_n
    //                     << " => "          << std::setw(7) << new_n
    //                     << " (x" << std::fixed << std::setprecision(2)
    //                     << (double)new_n / std::max(1, ori_n)
    //                     << "; thres=" << std::setprecision(4) << prune_thres << ")\n";

    //             std::cout << "[PRUNE:after] N=" << voxel_model_->numVoxels() << "\n";
    //             auto pr_dbg = voxel_model_->subdivisionPriority();
    //             std::cout << "    priority def=" << (pr_dbg.defined()?1:0)
    //                     << " shape=" << shp(pr_dbg)
    //                     << " numel=" << (pr_dbg.defined()? pr_dbg.numel() : 0) << "\n";
    //         }

    //         // SUBDIVIDE
    //         if (need_subdividing) {
    //             int ori_n = voxel_model_->numVoxels();

    //             // Exclude newly pruned ones from min_samp_interval if we just pruned
    //             auto min_samp_interval = stat.min_samp_interval; // [N,1]
    //             if (need_pruning && prune_mask.defined()) {
    //                 auto keep_mask = (~prune_mask).to(torch::kBool);
    //                 min_samp_interval = min_samp_interval.index({keep_mask});
    //             }

    //             auto size_thres    = min_samp_interval * opt_params_.subdivide_samp_thres_; // [M,1]
    //             auto vox_size      = voxel_model_->voxSize();                                // [M,1]
    //             auto large_enough  = (vox_size * 0.5 > size_thres).squeeze(1);               // [M] bool
    //             // auto non_finest    = (voxel_model_->octLevel().squeeze(1).to(torch::kInt32)
    //             //                     < voxel_model_->maxNumLevels());                        // [M] bool
    //             auto octlv        = voxel_model_->octLevel();
    //             if (octlv.defined() && octlv.dim()==2 && octlv.size(1)==1) octlv = octlv.squeeze(1);
    //             auto non_finest   = (octlv.to(torch::kInt32) < voxel_model_->maxNumLevels()); // [M]
    //             auto valid_mask    = large_enough & non_finest;
    //             auto priority = voxel_model_->subdivisionPriority();              // [M]

    //             std::cout << "[SUBDIV:prep] N=" << voxel_model_->numVoxels()
    //                     << "  large_enough=" << shp(large_enough)
    //                     << "  octlevel=" << shp(voxel_model_->octLevel())
    //                     << "  octlv(1D)=" << shp(octlv)
    //                     << "  non_finest=" << shp(non_finest)
    //                     << "  valid_mask=" << shp(valid_mask)
    //                     << "  sum(valid)=" << (valid_mask.defined()? valid_mask.sum().item<int64_t>() : -1) << "\n";

    //             std::cout << "    priority def=" << (priority.defined()?1:0)
    //                     << " shape=" << shp(priority)
    //                     << " numel=" << (priority.defined()? priority.numel() : 0) << "\n";

    //             // --- the line that explodes if shapes differ:
    //             std::cout << "    CHECK: priority.size(0)="
    //                     << (priority.defined() && priority.dim()? priority.size(0) : -1)
    //                     << "  vs valid_mask.size(0)=" << (valid_mask.defined()? valid_mask.size(0) : -1) << "\n";

    //             priority = priority * valid_mask; // mask to zero for invalid

    //             torch::Tensor subdivide_mask;
    //             if (iter <= opt_params_.subdivide_all_until_) {
    //                 subdivide_mask = valid_mask; // take all valids early on
    //             } else {
    //                 // threshold by (1 - subdivide_prop_) quantile
    //                 double q = std::max(0.0, 1.0 - (double)opt_params_.subdivide_prop_);
    //                 auto thres = priority.quantile(q);
    //                 subdivide_mask = (priority > thres) & valid_mask;
    //             }

    //             // cap number of parents (each makes +7 children)
    //             int max_n_subdiv = std::round(
    //                 (opt_params_.subdivide_max_num_ - voxel_model_->numVoxels()) / 7.0);

    //             if (max_n_subdiv > 0) {
    //                 int num_sel = (int)subdivide_mask.sum().item<int64_t>();
    //                 if (num_sel > max_n_subdiv) {
    //                     auto pos_idx  = subdivide_mask.nonzero().squeeze(1); // [K]
    //                     auto pos_vals = priority.index({pos_idx});           // [K]
    //                     auto sorted   = std::get<0>(pos_vals.sort(/*dim=*/0)); // asc
    //                     int  n_removed = num_sel - max_n_subdiv;
    //                     auto cutoff    = sorted.index({n_removed - 1});
    //                     subdivide_mask = subdivide_mask & (priority > cutoff);
    //                 }

    //                 voxel_model_->subdividing(subdivide_mask);
    //                 int new_n = voxel_model_->numVoxels();

    //                 std::cout << "[SUBDIVIDING] " << std::setw(7) << ori_n
    //                         << " => "          << std::setw(7) << new_n
    //                         << " (x" << std::fixed << std::setprecision(2)
    //                         << (double)new_n / std::max(1, ori_n) << ")\n";

    //                 voxel_model_->resetSubdivisionPriority();
    //             }
    //         }

    //         // if (changed) {
    //         //     auto render_pkg2 = voxel_model_->render(cam);
    //         //     auto masked2 = render_pkg2["color"].to(mDevice) * mask;
    //         //     saveVoxelErrorHeatmap(cam, masked2.detach().contiguous(),
    //         //                         masked_gt.detach().contiguous(),
    //         //                         viewpoint_cam->fid_, (result_dir_ / "heatmaps").string());
    //         // }

    //         voxel_model_->createTrainer(
    //             opt_params_.geo_lr_,
    //             opt_params_.sh0_lr_,
    //             opt_params_.shs_lr_,
    //             opt_params_.optim_beta1_,
    //             opt_params_.optim_beta2_,
    //             opt_params_.optim_eps_,
    //             opt_params_.lr_decay_ckpt_,
    //             opt_params_.lr_decay_mult_
    //         );
    //         voxel_model_->schedulerLoadStateDict(sched_state);
    //         // Empty CUDA cache as SV does
    //         {
    //             py::gil_scoped_acquire gil;
    //             py::module_ torch_mod = py::module_::import("torch");
    //             torch_mod.attr("cuda").attr("empty_cache")();
    //         }
    //     }
    // }
    
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

            // //  // Add new points to the model
            //  if (initial_mapped_ && points.size() >= 30) {
            //     extendAABB_with_flat_xyz(aabb_min_, aabb_max_, points);  // points = std::vector<float>
            //     have_bounds_ = true;

            //      torch::NoGradGuard no_grad;
            //      std::unique_lock<std::mutex> lock_render(mutex_render_);

            //     // log_increase_batch_npy(result_dir_, points, colors, getIteration(), next_batch_index_);
            //     // ++next_batch_index_;
            //     //  std::cout << "first increasePcd" << std::endl;
            //     //  voxel_model_->increasePcd(points, colors, getIteration(), kfs_for_bounding);
            //     voxel_model_->increasePcd(points, colors, getIteration());
            //     // py::object sched_state = voxel_model_->schedulerStateDict();
            //     // voxel_model_->createTrainer(
            //     //                             opt_params_.geo_lr_,
            //     //                             opt_params_.sh0_lr_,
            //     //                             opt_params_.shs_lr_,
            //     //                             opt_params_.optim_beta1_,
            //     //                             opt_params_.optim_beta2_,
            //     //                             opt_params_.optim_eps_,
            //     //                             opt_params_.lr_decay_ckpt_,
            //     //                             opt_params_.lr_decay_mult_);
            //     // voxel_model_->schedulerLoadStateDict(sched_state);
            //  }

            // Only try to grow if model was initialized and we have any new points
            if (initial_mapped_ && !points.empty()) {
                // 1) Get current bound
                torch::Tensor sc  = voxel_model_->SceneCenter();  // [3] if set
                torch::Tensor se  = voxel_model_->SceneExtent();  // [1] if set

                // If the model hasn't been initialized yet (shouldn’t happen here), skip.
                if (sc.defined() && se.defined() && sc.numel()==3 && se.numel()==1) {
                    auto sc_cpu = sc.detach().to(torch::kCPU);
                    auto se_cpu = se.detach().to(torch::kCPU);

                    // scene_min = center - 0.5 * extent;  scene_max = center + 0.5 * extent
                    float cx = sc_cpu[0].item<float>();
                    float cy = sc_cpu[1].item<float>();
                    float cz = sc_cpu[2].item<float>();
                    float ex = se_cpu[0].item<float>() * 0.5f;
                    float minx = cx - ex, maxx = cx + ex;
                    float miny = cy - ex, maxy = cy + ex;
                    float minz = cz - ex, maxz = cz + ex;

                    const float eps = 1e-6f; // tiny numerical slack

                    // 2) Keep only points outside current AABB
                    std::vector<float> out_pts;
                    std::vector<float> out_cols;
                    out_pts.reserve(points.size());  // upper bound
                    out_cols.reserve(colors.size());

                    const size_t N = points.size() / 3;
                    for (size_t i = 0; i < N; ++i) {
                        float x = points[3*i + 0];
                        float y = points[3*i + 1];
                        float z = points[3*i + 2];

                        bool outside =
                            (x < minx - eps) || (x > maxx + eps) ||
                            (y < miny - eps) || (y > maxy + eps) ||
                            (z < minz - eps) || (z > maxz + eps);

                        if (outside) {
                            out_pts.push_back(x);
                            out_pts.push_back(y);
                            out_pts.push_back(z);

                            out_cols.push_back(colors[3*i + 0]);
                            out_cols.push_back(colors[3*i + 1]);
                            out_cols.push_back(colors[3*i + 2]);
                        }
                    }

                    // 3) Only grow if enough out-of-bounds points
                    const int MIN_OUTSIDE = 10; // your threshold
                    if ((int)(out_pts.size()/3) >= MIN_OUTSIDE) {
                        // Optional: ensure we don't move the 'min side' if you want perfect key stability
                        // (Counts of which side is violated; skip if it would force min to move)
                        // bool touches_min_side = (min of any coord < minX/Y/Z);
                        // if (touches_min_side) { /* decide: defer / tile / rebase */ }

                        torch::NoGradGuard no_grad;
                        std::unique_lock<std::mutex> lock_render(mutex_render_);
                        // 4) Actually insert *only* the outside points
                        voxel_model_->increasePcd(out_pts, out_cols, getIteration());
                        // // Snapshot scheduler/opt state before we rebuild
                        // py::object sched_state = voxel_model_->schedulerStateDict();
                        // {
                        //     std::unique_lock<std::mutex> lock(mutex_settings_);
                        //     voxel_model_->createTrainer(
                        //         opt_params_.geo_lr_,
                        //         opt_params_.sh0_lr_,
                        //         opt_params_.shs_lr_,
                        //         opt_params_.optim_beta1_,
                        //         opt_params_.optim_beta2_,
                        //         opt_params_.optim_eps_,
                        //         opt_params_.lr_decay_ckpt_,
                        //         opt_params_.lr_decay_mult_);
                        //     voxel_model_->schedulerLoadStateDict(sched_state);
                        // }
                    }
                }
            }

            if (kf_changed) {
                dumpKeyframesForProjectionFile(scene_->keyframes(), scene_->cameras_,
                                            result_dir_ / "proj_debug");
            }
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
            if (kf_changed) {
                dumpKeyframesForProjectionFile(scene_->keyframes(), scene_->cameras_,
                                            result_dir_ / "proj_debug");
            }
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
    auto& pose = std::get<2>(kf);
    // ─── Set its pose ───────────────────────────────────────────────────────
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>()
    );
    cv::Mat imgRGB_undistorted;
    // Camera
    sv::Camera& camera = scene_->cameras_.at(std::get<1>(kf));
    pkf->setCameraParams(camera);

    // Image (left if STEREO)
    cv::Mat imgRGB = std::get<3>(kf);
    camera.undistortImage(imgRGB, imgRGB_undistorted);
    pkf->original_image_ =
        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
    pkf->img_filename_ = std::get<8>(kf);
     
     // Add the new keyframe to the scene
    //  pkf->computeTransformTensors();
     scene_->addKeyframe(pkf, &kfid_shuffled_);
 
     // Give new keyframes times of use and add it to the training sliding window
     increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
     pkf->img_undist_ = imgRGB_undistorted;
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
    const std::string&           name_suffix)
{
    // std::cout << "[DEBUG] Starting renderAndRecordKeyframe for frame "
    //           << pkf->fid_ << '\n';

    /* ------------------------------------------------ 1. camera  */
    sv::MiniCam cam = pkf->toMiniCam();
    // cam.c2w = cam.c2w.contiguous().to(mDevice);   // make sure contiguous + on CUDA
    // cam.w2c = cam.w2c.contiguous().to(mDevice);
    /* ------------------------------------------------ 2. GT → NumPy  */
    // torch::Tensor chw_u8 = pkf->original_image_    // (3,H,W) in [0,1]
    //                         .mul(255.0f)
    //                         .clamp(0.0f, 255.0f)
    //                         .to(torch::kUInt8)            // <--- cast to U8
    //                         .cpu()
    //                         .contiguous();
    // torch::Tensor hwc_u8 = chw_u8.permute({1,2,0}).contiguous();  // (H,W,3)
    // // convert CHW→HWC uint8 numpy without any CUDA involvement
    // py::array rgb_numpy = sv::tensorToNumpyRGB(hwc_u8);
    /* ------------------------------------------------ 3. render   */
    auto t0 = std::chrono::steady_clock::now();
    auto render_pkg = voxel_model_->render(cam);
    auto t1 = std::chrono::steady_clock::now();
    render_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

    if (render_pkg.empty() || !render_pkg.count("color") || !render_pkg.at("color").defined()) {
        std::cout << "render pkg empty" << std::endl;
        return;
    }
    torch::Tensor rendered_image = render_pkg.at("color").to(mDevice);          // (1,3,H,W)

    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    // saveTensor(masked_image,     "masked_image",    "/home/dimitris/Photo-SLAM/debug", getIteration(), pkf->fid_);
    masked_image = masked_image.squeeze(0);   
    auto gt_image = pkf->original_image_;
    // saveTensor(gt_image,     "gt_image",    "/home/dimitris/Photo-SLAM/debug", getIteration(), pkf->fid_);

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();

    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    
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
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, render_time, image_dir, image_gt_dir, image_loss_dir);
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
    voxel_model_->savePly(ply_dir / "point_cloud.ply");
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
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<VoxelKeyframe> pkf = std::make_shared<VoxelKeyframe>();
    // pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        sv::Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        // pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[Mapper::renderFromPose]KeyFrame Camera not found!");
    }

    // // MiniCam helper already implemented in VoxelKeyframe
    // sv::MiniCam miniCam = pkf->toMiniCam();
    // miniCam.c2w = miniCam.c2w.to(device_type_);
    // miniCam.w2c = miniCam.w2c.to(device_type_);

    // auto chw_u8 = pkf->original_image_
    //                   .mul(255.0f).clamp(0.0f,255.0f)
    //                   .to(torch::kUInt8)
    //                   .cpu()
    //                   .contiguous();
    // auto hwc_u8 = chw_u8.permute({1,2,0}).contiguous();
    // py::array rgb_numpy = sv::tensorToNumpyRGB(hwc_u8);

    // 3) call into your voxel_renderer under the lock
    std::unordered_map<std::string, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock(mutex_render_);
        render_pkg = voxel_model_->render(pkf->toMiniCam());
    }

    // 4) extract the “rgb” tensor (batch of 1×3×H×W)
    if (!render_pkg.count("color") || !render_pkg["color"].defined()) {
        // if rendering failed, return a black image
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    }
    at::Tensor rgb = render_pkg["color"];          // (1,3,H,W)
    rgb = rgb.to(torch::kCPU);
    rgb = rgb.squeeze(0);                        // → (3,H,W)

    cv::imwrite("debug_rgb.png", tensor_utils::torchTensor2CvMat_Float32(rgb));
    std::cout << "[renderFromPose] rgb min/max = "
          << rgb.min().item<float>() << "/"
          << rgb.max().item<float>() << "\n";

    // 5) apply the appropriate undistort mask
    at::Tensor mask = main_vision
        ? viewer_main_undistort_mask_.at(pkf->camera_id_)
        : viewer_sub_undistort_mask_.at(pkf->camera_id_);
    // ensure mask is on CPU and broadcastable
    mask = mask.to(torch::kFloat32).to(rgb.device());
    at::Tensor masked = rgb * mask;             // (3,H,W)

    // 6) convert back to cv::Mat
    return tensor_utils::torchTensor2CvMat_Float32(masked);
}

// VoxelMapper::~VoxelMapper() {
//     // Explicitly reset any Python or Torch objects that may call Python at destruction
//     voxel_model_.reset();  // Deallocates all tensors and Python wrappers
//     mpSLAM.reset();
// }

 bool VoxelMapper::isKeepingTraining()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     return keep_training_;
 }

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

int VoxelMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}

float VoxelMapper::lambdaDssim()
 {
     std::unique_lock<std::mutex> lock(mutex_settings_);
     return opt_params_.lambda_dssim_;
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