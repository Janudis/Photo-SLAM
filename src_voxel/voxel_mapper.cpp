#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_trainer.h"
#include "include_voxel/mini_cam.h"
#include "include_voxel/py_utils.h"
#include "include_voxel/voxel_constants.h"
#include "include_voxel/voxel_config.h"
#include "include/loss_utils.h"
#include "include_voxel/voxel_keyframe.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>     //  ← for gil_scoped_release

#include <opencv2/opencv.hpp>
#include <MapPoint.h>
#include <Map.h>
#include <fstream>
#include <sstream>
#include <set>
#include <memory>   // make sure this is at the top of the file

namespace py = pybind11;

#define READ_OR_EXIT(node, field, target)                           \
    if (!node[field].empty()) node[field] >> target;               \
    else { std::cerr << "[ERROR] Missing config key: " << field    \
                     << " in YAML.\n"; std::exit(EXIT_FAILURE); }
static sv::VoxelScheduleConfig loadVoxelConfig(const std::filesystem::path& yaml_path)
{
    sv::VoxelScheduleConfig cfg;

    cv::FileStorage fs(yaml_path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open voxel config: " << yaml_path << std::endl;
        std::exit(1);
    }

    READ_OR_EXIT(fs, "lr", cfg.lr);
    READ_OR_EXIT(fs, "prune_threshold_init", cfg.prune_threshold_init);
    READ_OR_EXIT(fs, "prune_threshold_final", cfg.prune_threshold_final);
    READ_OR_EXIT(fs, "subdiv_quantile", cfg.subdiv_quantile);
    READ_OR_EXIT(fs, "subdiv_gradient_threshold", cfg.subdiv_gradient_threshold);
    READ_OR_EXIT(fs, "min_voxels", cfg.min_voxels);
    READ_OR_EXIT(fs, "meta_warmup_iters", cfg.meta_warmup_iters);
    READ_OR_EXIT(fs, "subdiv_from", cfg.subdiv_from);
    READ_OR_EXIT(fs, "subdiv_every", cfg.subdiv_every);
    READ_OR_EXIT(fs, "subdiv_until", cfg.subdiv_until);
    READ_OR_EXIT(fs, "prune_from", cfg.prune_from);
    READ_OR_EXIT(fs, "prune_every", cfg.prune_every);
    READ_OR_EXIT(fs, "prune_until", cfg.prune_until);
    READ_OR_EXIT(fs, "meta_accum_lr", cfg.meta_accum_lr);

    if (!fs["loss_weights"].empty()) {
        cv::FileNode weights = fs["loss_weights"];
        if (!weights["photo"].empty())     weights["photo"]     >> cfg.lambda_photo;
        if (!weights["ssim"].empty())      weights["ssim"]      >> cfg.lambda_ssim;
        if (!weights["T_concen"].empty())  weights["T_concen"]  >> cfg.lambda_T_concen;
        if (!weights["T_inside"].empty())  weights["T_inside"]  >> cfg.lambda_T_inside;
    }

    return cfg;
}

//------------------------------------------------------------
// helper: copy Mat → NumPy (NumPy owns the memory)
//------------------------------------------------------------
static py::array_t<uint8_t> cvMatToNumpyRGB(const cv::Mat &img)
{
    if (img.empty() || img.type() != CV_8UC3)
        throw std::runtime_error("Expected a non‑empty 3‑channel CV_8UC3 image");

    // allocate new NumPy array that *owns* its data
    py::array_t<uint8_t> arr({img.rows, img.cols, 3});
    std::memcpy(arr.mutable_data(), img.data,
                static_cast<size_t>(img.rows * img.cols * 3));
    return arr;   // safe – Python holds the buffer
}

VoxelMapper::VoxelMapper(std::shared_ptr<ORB_SLAM3::System> slam,
            // const sv::VoxelScheduleConfig& cfg,
            const std::filesystem::path& voxel_yaml,
            const std::filesystem::path& seq_dir,
            const std::filesystem::path& out_dir,
            torch::Device device)
    :   mpSLAM(slam),
        // mVoxelConfig(cfg),
        mVoxelCfg(voxel_yaml),
        mSeqDir(seq_dir),
        mOutDir(out_dir),
        mDevice(device)
{
    // Load config from YAML
    mVoxelConfig = loadVoxelConfig(voxel_yaml);
    // Create trainer
    mpTrainer = std::make_shared<sv::VoxelTrainer>(64);  // grid resolution
}

void VoxelMapper::LoadImages(const std::string& rgb_txt_path)
{
    std::ifstream f(rgb_txt_path);
    std::string line;
    // Temp list to sort
    std::vector<std::pair<double, std::string>> entries;

    while (std::getline(f, line)) {
        if (line.empty() || line[0]=='#') continue;
        std::stringstream ss(line);
        double t;
        std::string fname;
        ss >> t >> fname;
        entries.emplace_back(t, (mSeqDir / fname).string());
    }
    // Ensure timestamps are sorted!
    std::sort(entries.begin(), entries.end());
    for (const auto& e : entries) {
        mTimestamps.push_back(e.first);
        mImagePaths.push_back(e.second);
    }
}

void VoxelMapper::run()
{
    py::gil_scoped_acquire gil;        // ← take the interpreter lock first
    // make our helper modules visible
    py::module_::import("sys")
        .attr("path").attr("insert")(0, "../scripts_voxel");
    // std::fprintf(stderr, "[DBG]  current interp = %p\n", PyThreadState_Get()->interp);
    LoadImages((mSeqDir / "rgb.txt").string());
 
    // Phase 1: Feed all images to SLAM
    size_t kf_index = 0;          // <- new
    for (size_t i = 0; i < mImagePaths.size(); ++i) 
    // for (size_t i = 0; i < mKeyframeIds.size(); ++i)
    {
        cv::Mat im = cv::imread(mImagePaths[i], cv::IMREAD_UNCHANGED);
        if (im.empty()) continue;
        if (im.type() == CV_16U) {
            double minV, maxV; cv::minMaxLoc(im, &minV, &maxV);
            im.convertTo(im, CV_8U, 255.0 / maxV);
        }
        cv::Mat imGray;
        if (im.channels() == 3) cv::cvtColor(im, imGray, cv::COLOR_BGR2GRAY);
        else                    imGray = im;

        double dt = (i == 0) ? 0.033 : (mTimestamps[i] - mTimestamps[i - 1]);
        if (dt > 0.0) std::this_thread::sleep_for(std::chrono::duration<double>(dt));

        Sophus::SE3f Tcw = mpSLAM->TrackMonocular(imGray, mTimestamps[i]);
        mTcwList.push_back(Tcw);
        mTrackState.push_back(mpSLAM->GetTrackingState());
    }

    // Phase 2: Check if SLAM initialized properly
    int n_kfs = mpSLAM->GetNumKeyframes();
    if (n_kfs < 15) {
        std::cout << "[ERROR] Not enough keyframes. SLAM failed to initialize.\n";
        return;
    }
    std::vector<ORB_SLAM3::KeyFrame*> keyframes = mpSLAM->getAtlas()->GetAllKeyFrames();
    std::sort(keyframes.begin(), keyframes.end(),
            [](ORB_SLAM3::KeyFrame* a, ORB_SLAM3::KeyFrame* b) {
                return a->mTimeStamp < b->mTimeStamp;
            });

    for (auto* kf : keyframes)
    {
        if (kf && !kf->isBad())
        {
            mKeyframeIds.push_back(kf->mnId);
            mKeyframePoses.push_back(Sophus::SE3f(kf->GetPose()));
            // Adjust this line if your filenames are different (e.g. .jpg)
            mKeyframeImages.push_back((mSeqDir / "rgb" / (std::to_string(kf->mTimeStamp) + ".png")).string());
        }
    }
    // === Create VoxelKeyframes ===
    for (size_t i = 0; i < mKeyframeIds.size(); ++i) {
        auto kf = std::make_shared<VoxelKeyframe>();
        kf->fid_ = mKeyframeIds[i];
        kf->img_path_ = mKeyframeImages[i];
        kf->Tcw = mKeyframePoses[i];
        kf->remaining_times_of_use_ = 1;
        mSceneKeyframes[kf->fid_] = kf;
    }

    // Phase 3: Initialize voxels from SLAM map
    float voxel_size = 0.05f;
    torch::Tensor voxel_centers;
    if (!initializeVoxelsFromMap(voxel_centers, voxel_size)) {
        std::cout << "[ERROR] No valid map points after SLAM.\n";
        return;
    }

    const int64_t N = voxel_centers.size(0);
    torch::Tensor oct_paths = torch::arange(N, torch::kLong).to(mDevice);
    torch::Tensor oct_levels  = torch::zeros({N},  torch::kInt32 ).to(mDevice);
    // torch::Tensor subdiv_meta = torch::zeros({N},  torch::kFloat32).to(mDevice);
    torch::Tensor subdiv_meta = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(mDevice).requires_grad(true));
    subdiv_meta.retain_grad();  // ← THIS is crucial for non-leaf tensors
    torch::Tensor subdiv_p = torch::zeros_like(subdiv_meta, torch::kFloat32).to(mDevice);


    mpTrainer->set_voxels(
            voxel_centers.to(mDevice),                           // (N,3)
            torch::full({N},   voxel_size, torch::kFloat32).to(mDevice),      // lengths
            torch::zeros({N,8},  torch::kFloat32).to(mDevice),   // geo key‑pts (unused)
            torch::ones ({N,3},  torch::kFloat32).to(mDevice)*0.5f,           // RGB
            torch::zeros({N,45}, torch::kFloat32).to(mDevice),   // SH coeffs
            torch::ones ({N},    torch::kFloat32).to(mDevice)*0.8f,           // opacity
            oct_paths,
            oct_levels,
            subdiv_meta,
            subdiv_p                                            
    );
    std::cout << "[INFO] Voxel model initialized from SLAM map.\n";

    // Phase 4: Train on all frames again with optimization
    trainLoopWithOptimization(10000);  // or any number of iterations you prefer
    finalize();
    mpTrainer.reset();
}

bool VoxelMapper::initializeVoxelsFromMap(torch::Tensor& out_centers, float voxel_size)
{
    auto map = mpSLAM->getAtlas()->GetCurrentMap();
    std::vector<ORB_SLAM3::MapPoint*> mps = map->GetAllMapPoints();
    std::set<std::vector<float>> unique_centers;

    for (auto* mp : mps) {
        if (!mp || mp->isBad()) continue;
        Eigen::Vector3f pos = mp->GetWorldPos().cast<float>();
        Eigen::Vector3f center = (pos / voxel_size).array().round() * voxel_size;
        unique_centers.insert({center[0], center[1], center[2]});
    }

    if (unique_centers.empty()) return false;

    std::cout << "[INFO] Loaded " << unique_centers.size() << " unique voxel centers.\n";

    std::vector<torch::Tensor> tensor_list;
    for (const auto& vec : unique_centers) {
        tensor_list.push_back(torch::from_blob((void*)vec.data(), {3}, torch::kFloat32).clone());
    }

    out_centers = torch::stack(tensor_list);
    return true;
}

//------------------------------------------------------------------
// Save a single RGB tensor to disk (expects H×W×3 uint8 NumPy)
//------------------------------------------------------------------
static void saveNumpyRGB(const py::array &arr,
                         const std::filesystem::path &png_path)
{
    cv::Mat img(arr.shape(0),               // rows (H)
                arr.shape(1),               // cols (W)
                CV_8UC3,
                (void*)arr.data());  
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imwrite(png_path.string(), img);
}

//------------------------------------------------------------------
// Render every cached key‑frame once and write to …/NNNN_shutdown/
//------------------------------------------------------------------
void VoxelMapper::renderAndDumpAllKeyframes(const std::string &tag)
{
    const auto dump_root = mOutDir / (tag + "_shutdown");
    const auto img_dir   = dump_root / "image";
    std::filesystem::create_directories(img_dir);

    std::ofstream timings(dump_root / "render_time.txt",
                          std::ios::out | std::ios::trunc);
    timings << "## keyframe_id   time_ms\n";

    for (size_t i = 0; i < mKeyframeIds.size(); ++i)
    {
        int kfid = mKeyframeIds[i];
        const Sophus::SE3f& Tcw = mKeyframePoses[i];
        const std::string& path = mKeyframeImages[i];

        cv::Mat im = cv::imread(path, cv::IMREAD_COLOR);
        if (im.empty()) continue;

        sv::MiniCam cam;
        cam.width   = im.cols;    cam.height = im.rows;
        cam.cx      = cam.width * 0.5f;  cam.cy = cam.height * 0.5f;
        cam.fovx    = 2.f * std::atan(cam.width  / (2.f * 517.306f));
        cam.fovy    = 2.f * std::atan(cam.height / (2.f * 516.469f));
        cam.frame_id = kfid;

        Eigen::Matrix4f eig_Tcw = Tcw.matrix();
        Eigen::Matrix4f eig_c2w = eig_Tcw.inverse();

        cam.c2w = torch::from_blob(eig_c2w.data(), {4,4}, torch::kFloat32).clone().to(mDevice);
        cam.w2c = torch::from_blob(eig_Tcw.data(), {4,4}, torch::kFloat32).clone().to(mDevice);

        cv::Mat imRGB;
        cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);
        py::array rgb_numpy = cvMatToNumpyRGB(imRGB);

        auto t0 = std::chrono::high_resolution_clock::now();
        // std::cout << "[DBG] Rendering frame " << i << " with " << mpTrainer->num_voxels() << " voxels\n";
        auto out_map = mpTrainer->render(cam, rgb_numpy, img_dir.string());
        at::Tensor rgb_tensor_f32 = out_map.at("rgb").cpu();                 // (3,H,W) float32
        // at::Tensor rgb_tensor_u8  = (rgb_tensor_f32.clamp(0,1) * 255.0f)
        //                                 .to(torch::kU8)
        //                                 .permute({1,2,0})        // → (H,W,3) for OpenCV
        //                                 .contiguous();
        at::Tensor rgb_tensor_u8 =  (rgb_tensor_f32.squeeze(0)          //  (3,H,W)
                                        .clamp(0,1) * 255.0f)
                                        .to(torch::kU8)
                                        .permute({1,2,0})               //  (H,W,3) ok
                                        .contiguous();

        auto t1 = std::chrono::high_resolution_clock::now();
        std::stringstream ss;
        ss << tag << "_" << kfid << ".jpg";

        cv::Mat rgb_image(rgb_tensor_u8.size(0),               // rows = H
                        rgb_tensor_u8.size(1),               // cols = W
                        CV_8UC3,
                        rgb_tensor_u8.data_ptr());
        cv::cvtColor(rgb_image, rgb_image, cv::COLOR_RGB2BGR);
        cv::imwrite((img_dir / ss.str()).string(), rgb_image);

        double t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        timings << kfid << "  " << std::fixed << std::setprecision(3) << t_ms << "\n";
    }
}

void VoxelMapper::generateKfidRandomShuffle() {
    if (mSceneKeyframes.empty()) return;

    size_t nkfs = mSceneKeyframes.size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(std::random_device{}());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);
    kfid_shuffled_ = true;
}

std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomSlidingWindowKeyframe() {
    if (mSceneKeyframes.empty()) return nullptr;
    if (!kfid_shuffled_) generateKfidRandomShuffle();

    int usable = 0;
    for (const auto& [id, kf] : mSceneKeyframes) {
        if (kf->remaining_times_of_use_ > 0) ++usable;
    }
    std::cout << "[DEBUG] Keyframes with remaining_times_of_use_ > 0: " << usable << std::endl;

    const int max_attempts = static_cast<int>(mSceneKeyframes.size()) * 2;
    int attempts = 0;

    while (attempts++ < max_attempts)
    {
        if (kfid_shuffle_idx_ >= static_cast<int>(kfid_shuffle_.size()))
            kfid_shuffle_idx_ = 0;

        int idx = kfid_shuffle_[kfid_shuffle_idx_++];
        auto it = mSceneKeyframes.begin();
        std::advance(it, idx);
        auto viewpoint_cam = it->second;

        if (viewpoint_cam->remaining_times_of_use_ > 0) {
            --(viewpoint_cam->remaining_times_of_use_);
            kfs_used_times_[viewpoint_cam->fid_]++;
            return viewpoint_cam;
        }
    }

    // 🧨 All keyframes exhausted. Reset with fixed count!
    std::cout << "[INFO] Reset keyframe usage counts.\n";
    for (auto& kfit : mSceneKeyframes) {
        kfit.second->remaining_times_of_use_ = 1;   // ← Set explicitly
    }

    kfid_shuffle_idx_ = 0;
    return useOneRandomSlidingWindowKeyframe();
}

void VoxelMapper::increaseKeyframeTimesOfUse(std::shared_ptr<VoxelKeyframe> pkf, int times) {
    pkf->remaining_times_of_use_ += times;
}

// Add training with optimization loop
void VoxelMapper::trainLoopWithOptimization(int num_iters)
{
    auto optimizer = std::make_unique<torch::optim::Adam>(
        mpTrainer->parameters(), torch::optim::AdamOptions(mVoxelConfig.lr));
    bool changed = false;  // tracks if subdivide/prune occurred

    // Extract loss weights from config
    float w_photo    = mVoxelConfig.lambda_photo;
    float w_ssim     = mVoxelConfig.lambda_ssim;
    float w_tconcen  = mVoxelConfig.lambda_T_concen;
    float w_tinside  = mVoxelConfig.lambda_T_inside;

    for (int iter = 0; iter < num_iters; ++iter)
    {
        int total_kfs = mSceneKeyframes.size();
        std::cout << "[INFO] Iter " << iter << " | Total available keyframes: " << total_kfs << std::endl;

        float total_loss = 0.f;
        int frames_used = 0;

        // for (size_t i = 0; i < mImagePaths.size(); ++i)
        // {
            // if (mTrackState[i] != 1) continue;
            // std::cout << "[INFO] Using keyframe index " << i
            //    << " (path = " << mImagePaths[i] << ")\n";
            // cv::Mat im = cv::imread(mImagePaths[i], cv::IMREAD_COLOR);

        std::shared_ptr<VoxelKeyframe> pkf = useOneRandomSlidingWindowKeyframe();
        if (!pkf) continue;
        std::cout << "[INFO] Using keyframe index " << pkf->fid_
                << " (path = " << pkf->img_path_ << ")\n";
        std::cout << "[INFO] Remaining uses for keyframe " << pkf->fid_
                << ": " << pkf->remaining_times_of_use_ << std::endl;
        cv::Mat im = cv::imread(pkf->img_path_, cv::IMREAD_COLOR);

        if (im.empty()) continue;
        cv::Mat imRGB;
        cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);
        py::array rgb_numpy = cvMatToNumpyRGB(imRGB);

        sv::MiniCam cam;
        cam.width = im.cols; cam.height = im.rows;
        cam.cx = cam.width * 0.5f; cam.cy = cam.height * 0.5f;
        cam.fovx = 2.f * std::atan(cam.width  / (2.f * 517.306f));
        cam.fovy = 2.f * std::atan(cam.height / (2.f * 516.469f));
        // cam.frame_id = static_cast<int>(i);
        cam.frame_id = static_cast<int>(pkf->fid_);

        // Eigen::Matrix4f Tcw = mTcwList[i].matrix();
        Eigen::Matrix4f Tcw = pkf->Tcw.matrix();  // ← replace with your actual pose logic
        Eigen::Matrix4f c2w = Tcw.inverse();
        cam.c2w = torch::from_blob(c2w.data(), {4,4}, torch::kFloat32).clone().to(mDevice);
        cam.w2c = torch::from_blob(Tcw.data(), {4,4}, torch::kFloat32).clone().to(mDevice);

        torch::Tensor gt = torch::from_blob(imRGB.data, {cam.height, cam.width, 3}, torch::kUInt8)
                                .permute({2, 0, 1})
                                .to(torch::kFloat32)
                                .div(255.0f)
                                .to(mDevice)
                                .clone();
        try {
            auto subdiv_p = mpTrainer->get_tensor("subdiv_p");
            // std::cout << "[DEBUG] subdiv_p.requires_grad(): " << (subdiv_p.requires_grad() ? "true" : "false") << std::endl;
            auto out_map = mpTrainer->render(cam, rgb_numpy, "");
            torch::Tensor pred = out_map["rgb"].to(mDevice);
            // torch::Tensor gt_tensor = out_map["gt"].to(mDevice);  // shape (1,3,H,W)
            torch::Tensor gt_tensor = gt.unsqueeze(0);   // (1,3,H,W)
            
            TORCH_CHECK(pred.sizes() == gt_tensor.sizes(), "pred and gt_tensor shapes must match");
                // drop the interpreter lock for the heavy autograd work
            py::gil_scoped_release no_gil;

            torch::Tensor l2   = torch::mse_loss(pred, gt_tensor);
            torch::Tensor loss = w_photo * l2;
            if (out_map.find("T") != out_map.end())
            {
                torch::Tensor T         = out_map["T"].to(mDevice);   // (1,1,H,W)
                torch::Tensor T_concen  = (T * (1.0f - T)).mean();
                torch::Tensor T_inside  = T.square().mean();
                loss += w_tconcen * T_concen + w_tinside * T_inside;
            }
            auto pred_contig = pred.contiguous();
            auto gt_contig   = gt_tensor.contiguous();
            // SSIM term
            // torch::Tensor ssim_loss = 1.0f - loss_utils::ssim(pred, gt_tensor, mDevice.type());
            auto ssim_loss   = 1.0f - loss_utils::ssim(pred_contig, gt_contig, mDevice.type());
            loss += w_ssim * ssim_loss;

            // Backward pass
            (*optimizer).zero_grad();
            loss.backward();
            (*optimizer).step();                // ← Applies optimizer step

            // === Accumulate gradient for subdivision ===
            try {
                torch::NoGradGuard no_grad;
                auto grad = mpTrainer->get_subdiv_priority_grad();   // ← Safe accessor
                // std::cout << "[DEBUG] subdiv_p.grad().max(): " << grad.max().item<float>() << std::endl;
                
                // auto grad_detached = grad.detach();
                // torch::Tensor subdiv_meta = mpTrainer->get_tensor("subdiv_meta").detach();  // ← no grad needed here
                // auto updated = subdiv_meta + grad_detached * mVoxelConfig.meta_accum_lr;

                auto grad_copy = grad.clone();
                // grab current subdiv_meta (still a leaf)
                auto subdiv_meta = mpTrainer->get_tensor("subdiv_meta");
                // std::cout << "subdiv_meta_.shape = " << mpTrainer->get_tensor("subdiv_meta").sizes() << "\n";
                // std::cout << "grad_copy.shape     = " << grad_copy.sizes() << "\n";
                // accumulate
                auto updated = subdiv_meta + grad_copy * mVoxelConfig.meta_accum_lr;

                updated = torch::where(torch::isfinite(updated), updated, torch::zeros_like(updated));
                updated = updated.clamp(0.0f, 1.0f);

                mpTrainer->set_subdiv_meta(updated);  // ← retains_grad again internally

                // === NEW: Accumulate for gradient-based subdivision ===
                torch::Tensor all_indices = torch::arange(grad.numel(), grad.options().dtype(torch::kLong));
                mpTrainer->accumulate_subdiv_gradients(all_indices, grad_copy); 

            } catch (const std::exception& e) {
                std::cerr << "[WARN] Could not accumulate subdivision gradients: " << e.what() << std::endl;
            }

            total_loss += loss.item<float>();
            frames_used++;

        } catch (const py::error_already_set& e) {
            std::cerr << "[Python-Error] " << e.what() << std::endl;
            continue;
        }
        // }
        
        const int subdiv_every = 5;
        if ((iter + 1) % mVoxelConfig.subdiv_every == 0)
        {
            if (iter >= mVoxelConfig.subdiv_from && iter <= mVoxelConfig.subdiv_until) {
                torch::Tensor grad;
                try {
                    grad = mpTrainer->get_subdiv_priority_grad();  // ← uses subdiv_p_.grad()
                } catch (const std::exception& e) {
                    std::cerr << "[WARN] Cannot get subdivision gradients: " << e.what() << "\n";
                    return;
                }

                if (!grad.defined() || grad.numel() == 0) return;

                torch::Tensor grad_detached = grad.detach();

                if (!grad_detached.isfinite().all().item<bool>()) {
                    std::cerr << "[WARN] subdiv_p.grad contains NaNs or infs. Skipping...\n";
                    return;
                }

                float thresh = grad_detached.quantile(mVoxelConfig.subdiv_quantile).item<float>();

                auto size_ten = mpTrainer->get_tensor("size");
                torch::Tensor can_split = (size_ten * 0.5) >= sv::MIN_VOX_SIZE;

                torch::Tensor valid_mask = mpTrainer->get_tensor("oct_level") < sv::MAX_VOXEL_LEVEL;

                torch::Tensor mask = (grad_detached > thresh)
                                & (grad_detached > mVoxelConfig.subdiv_gradient_threshold)
                                & can_split
                                & valid_mask;

                int64_t n_sub = mask.sum().item<int64_t>();
                // std::cout << "[DBG] Subdivision candidates: " << n_sub << "\n";

                if (n_sub > 0) {
                    std::cout << "[SUBDIV] iter " << iter << "  |  splitting " << n_sub << " voxels\n";
                    mpTrainer->subdivide(mask);  // also resets subdiv_p_ internally
                    auto new_meta = torch::zeros_like(mpTrainer->get_tensor("subdiv_p"));  // now subdiv_p has correct shape
                    mpTrainer->set_subdiv_meta(new_meta);  // this will call .retain_grad() as needed

                    changed = true;
                    std::cout << "[INFO] Voxels after subdivide: " << mpTrainer->num_voxels() << "\n";
                }
            }
        }
        // === Pruning ===
        int64_t before = 0;
        int64_t after  = 0;
        torch::Tensor keep_mask;

        if ((iter >= mVoxelConfig.prune_from) &&
            (iter <= mVoxelConfig.prune_until) &&
            (iter % mVoxelConfig.prune_every == 0)) {

            torch::NoGradGuard _;

            auto size_ten  = mpTrainer->get_tensor("size");
            auto meta_ten  = mpTrainer->get_tensor("subdiv_meta");

            // Adaptive prune threshold
            float prune_iter_rate = float(iter - mVoxelConfig.prune_from) /
                                    float(mVoxelConfig.prune_until - mVoxelConfig.prune_from);
            prune_iter_rate = std::clamp(prune_iter_rate, 0.f, 1.f);

            float thresh = mVoxelConfig.prune_threshold_init +
                        (mVoxelConfig.prune_threshold_final - mVoxelConfig.prune_threshold_init) * prune_iter_rate;

            // torch::Tensor too_small = size_ten < sv::MIN_VOX_SIZE;
            // torch::Tensor low_meta  = meta_ten < thresh;
            // torch::Tensor keep_mask = ~(too_small | low_meta);

            // 1. Remove invalid values (NaNs or infs)
            meta_ten = torch::where(torch::isfinite(meta_ten), meta_ten, torch::zeros_like(meta_ten));
            // 2. Compute adaptive threshold via quantile (SVRaster-style)
            float quant = meta_ten.quantile(mVoxelConfig.subdiv_quantile).item<float>();
            // 3. Create mask: keep voxels with meta >= quantile threshold
            torch::Tensor keep_mask = (meta_ten >= quant) & (size_ten >= sv::MIN_VOX_SIZE);

            if (keep_mask.sum().item<int64_t>() < mVoxelConfig.min_voxels)
                keep_mask.slice(0, 0, mVoxelConfig.min_voxels).fill_(true);

            int64_t before = mpTrainer->num_voxels();
            int64_t after  = keep_mask.sum().item<int64_t>();

            if (after < before) {
                mpTrainer->prune(keep_mask);
                std::cout << "[PRUNE] " << before << " → " << after << " voxels\n";
                changed = true;
            }

            std::cout << "[DEBUG] After pruning: total_loss = " << total_loss
                    << " | frames_used = " << frames_used
                    << " | average = " << (frames_used > 0 ? total_loss / frames_used : 0.0f)
                    << std::endl;

            meta_ten.detach_().zero_();  // reset stats
        }
        if (changed) {
            // optimizer = build_optimizer();   // grab fresh leaf tensors
            optimizer = std::make_unique<torch::optim::Adam>(
                mpTrainer->parameters(), torch::optim::AdamOptions(mVoxelConfig.lr));
            std::cout << "[INFO] Rebuilt optimizer after topology change\n";
            changed = false;
        }   
        // std::cout << "[TRAIN] Iter " << iter << "  |  Loss: "
        //           << (frames_used > 0 ? total_loss / frames_used : 0.0) << std::endl;
        std::cout << "[TRAIN] Iter " << iter
          << " | Used " << frames_used << " keyframes out of " << total_kfs
          << " | Avg Loss: " << (frames_used > 0 ? total_loss / frames_used : 0.0)
          << std::endl;
    }
}

void VoxelMapper::finalize()
{
    std::filesystem::create_directories(mOutDir);
    // auto model_path = mOutDir / "voxels.pth";
    // mpTrainer->save_torch(model_path);
    // Match GaussianMapper naming
    std::string tag = std::to_string(static_cast<int>(mImagePaths.size()));
    renderAndDumpAllKeyframes(tag);
    // Match trajectory/timing output of GaussianMapper
    const std::string out_dir_str = mOutDir.string();
    mpSLAM->SaveTrajectoryTUM(out_dir_str + "/CameraTrajectory_TUM.txt");
    mpSLAM->SaveKeyFrameTrajectoryTUM(out_dir_str + "/KeyFrameTrajectory_TUM.txt");
    mpSLAM->SaveTrajectoryEuRoC(out_dir_str + "/CameraTrajectory_EuRoC.txt");
    mpSLAM->SaveKeyFrameTrajectoryEuRoC(out_dir_str + "/KeyFrameTrajectory_EuRoC.txt");
    // mpSLAM->SaveTrackingTimes(out_dir_str + "/TrackingTime.txt");
    // mpSLAM->SaveGpuTime(out_dir_str + "/GpuPeakUsageMB.txt");
}

VoxelMapper::~VoxelMapper() {
    // Explicitly reset any Python or Torch objects that may call Python at destruction
    mpTrainer.reset();  // Deallocates all tensors and Python wrappers
    mpSLAM.reset();

    mKeyframeImages.clear();
    mKeyframeIds.clear();
    mKeyframePoses.clear();
    mTcwList.clear();
    mImagePaths.clear();
    mTimestamps.clear();
}
