#include "include_voxel/voxel_mapper.h"

namespace py = pybind11;

VoxelMapper::VoxelMapper(std::shared_ptr<ORB_SLAM3::System> pSLAM,
                         const std::filesystem::path& config_file_path,
                         const std::filesystem::path& seq_dir,
                         const std::filesystem::path& out_dir,
                         torch::DeviceType device_type,
                         int seed)
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

    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[VoxelMapper] CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        mDevice = torch::Device(torch::kCUDA);
    } else {
        std::cout << "[VoxelMapper] Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        mDevice = torch::Device(torch::kCPU);
    }

    // Store paths
    config_file_path_ = config_file_path;
    mSeqDir = seq_dir;
    mOutDir = out_dir;

    // Create output directory
    result_dir_ = mOutDir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir_);

    // Load YAML settings
    readConfigFromFile(config_file_path_);

    // Create trainer with grid resolution
    mpTrainer = std::make_shared<sv::VoxelTrainer>(64);  // or mVoxelConfig.resolution

    // Background & override color setup
    std::vector<float> bg_color = {0.0f, 0.0f, 0.0f};  // no white-background logic needed
    background_ = torch::tensor(bg_color,
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

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

    // === Load main SLAM camera ===
    auto vpCameras = mpSLAM->getAtlas()->GetAllCameras();
    if (!vpCameras.empty()) {
        mpCamera = vpCameras.front();  // ← assuming monocular, like Photo-SLAM
        std::cout << "[VoxelMapper] Loaded SLAM camera with id: " << mpCamera->GetId() << std::endl;
    } else {
        std::cerr << "[VoxelMapper] Error: No cameras found in SLAM atlas!" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void VoxelMapper::readConfigFromFile(const std::filesystem::path& cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string(), cv::FileStorage::READ);
    if (!settings_file.isOpened()) {
        std::cerr << "[VoxelMapper] Failed to open cfg: "
                  << cfg_path << '\n';
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[VoxelMapper] Reading parameters from "
              << cfg_path << '\n';
    std::lock_guard<std::mutex> guard(mutex_status_);

    /* ───────── OPTIMISATION ───────── */
    mVoxelConfig.lr                    = settings_file["Optimization.lr"];
    mVoxelConfig.meta_accum_lr         = settings_file["Optimization.meta_accum_lr"];

    mVoxelConfig.subdiv_from           = settings_file["Optimization.subdiv_from"];
    mVoxelConfig.subdiv_every          = settings_file["Optimization.subdiv_every"];
    mVoxelConfig.subdiv_until          = settings_file["Optimization.subdiv_until"];
    mVoxelConfig.subdiv_quantile       = settings_file["Optimization.subdiv_quantile"];
    mVoxelConfig.subdiv_gradient_threshold
                                       = settings_file["Optimization.subdiv_gradient_threshold"];

    mVoxelConfig.prune_from            = settings_file["Optimization.prune_from"];
    mVoxelConfig.prune_every           = settings_file["Optimization.prune_every"];
    mVoxelConfig.prune_until           = settings_file["Optimization.prune_until"];
    mVoxelConfig.prune_threshold_init  = settings_file["Optimization.prune_threshold_init"];
    mVoxelConfig.prune_threshold_final = settings_file["Optimization.prune_threshold_final"];
    mVoxelConfig.min_voxels            = settings_file["Optimization.min_voxels"];

    mVoxelConfig.max_num_iterations    = settings_file["Optimization.max_num_iterations"];
    mVoxelConfig.densification_interval_
                                       = settings_file["Optimization.densification_interval"];

    /* position-LR schedule */
    lr_init_           = settings_file["Optimization.position_lr_init"];
    pos_lr_delay_mult_ = settings_file["Optimization.position_lr_delay_mult"];
    pos_lr_max_steps_  = settings_file["Optimization.position_lr_max_steps"];

    /* ───────── PIPELINE FLAGS ───────── */
    do_inactive_geo_densify_
        = (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;

    var_params_.do_inactive_geo_densify      = do_inactive_geo_densify_;
    var_params_.new_keyframe_times_of_use    =
        settings_file["Mapper.new_keyframe_times_of_use"];
    min_num_initial_map_kfs_ = static_cast<std::size_t>(
        settings_file["Mapper.min_num_initial_map_kfs"].operator int());

    large_rot_th_   = settings_file["Mapper.large_rotation_threshold"];
    large_trans_th_ = settings_file["Mapper.large_translation_threshold"];
    cull_keyframes_ = (settings_file["Mapper.cull_keyframes"].operator int()) != 0;

    /* ───────── LOGGING ───────── */
    training_report_interval_      = settings_file["Record.training_report_interval"];
    keyframe_record_interval_      = settings_file["Record.keyframe_record_interval"];
    all_keyframes_record_interval_ = settings_file["Record.all_keyframes_record_interval"];

    record_rendered_image_    = (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_= (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_        = (settings_file["Record.record_loss_image"].operator int()) != 0;

    mVoxelConfig.lambda_photo    = 1.f;
    mVoxelConfig.lambda_ssim     = 0.02f;

    var_params_.lambda_photo    = mVoxelConfig.lambda_photo;
    var_params_.lambda_dssim    = mVoxelConfig.lambda_ssim;

    std::cout << "\n[CFG] Parsed Optimization Parameters:" << std::endl;
    std::cout << "  lr:                       " << mVoxelConfig.lr << std::endl;
    std::cout << "  meta_accum_lr:           " << mVoxelConfig.meta_accum_lr << std::endl;
    std::cout << "  position_lr_init:        " << lr_init_ << std::endl;
    std::cout << "  position_lr_delay_mult:  " << pos_lr_delay_mult_ << std::endl;
    std::cout << "  position_lr_max_steps:   " << pos_lr_max_steps_ << std::endl;
    std::cout << "  max_num_iterations:      " << mVoxelConfig.max_num_iterations << std::endl;
    std::cout << "  densification_interval:  " << mVoxelConfig.densification_interval_ << std::endl;

    std::cout << "\n[CFG] Subdivision Parameters:" << std::endl;
    std::cout << "  subdiv_from:             " << mVoxelConfig.subdiv_from << std::endl;
    std::cout << "  subdiv_every:            " << mVoxelConfig.subdiv_every << std::endl;
    std::cout << "  subdiv_until:            " << mVoxelConfig.subdiv_until << std::endl;
    std::cout << "  subdiv_quantile:         " << mVoxelConfig.subdiv_quantile << std::endl;
    std::cout << "  subdiv_gradient_threshold: " << mVoxelConfig.subdiv_gradient_threshold << std::endl;

    std::cout << "\n[CFG] Pruning Parameters:" << std::endl;
    std::cout << "  prune_from:              " << mVoxelConfig.prune_from << std::endl;
    std::cout << "  prune_every:             " << mVoxelConfig.prune_every << std::endl;
    std::cout << "  prune_until:             " << mVoxelConfig.prune_until << std::endl;
    std::cout << "  prune_threshold_init:    " << mVoxelConfig.prune_threshold_init << std::endl;
    std::cout << "  prune_threshold_final:   " << mVoxelConfig.prune_threshold_final << std::endl;
    std::cout << "  min_voxels:              " << mVoxelConfig.min_voxels << std::endl;

    std::cout << "\n[CFG] Loss Weights (hardcoded for now):" << std::endl;
    std::cout << "  lambda_photo:            " << mVoxelConfig.lambda_photo << std::endl;
    std::cout << "  lambda_ssim:             " << mVoxelConfig.lambda_ssim << std::endl;
    // std::cout << "  lambda_T_concen:         " << mVoxelConfig.lambda_T_concen << std::endl;
    std::cout << "  lambda_T_inside:         " << mVoxelConfig.lambda_T_inside << std::endl;

    std::cout << "\n[CFG] Pipeline & Mapper Flags:" << std::endl;
    std::cout << "  inactive_geo_densify:    " << do_inactive_geo_densify_ << std::endl;
    std::cout << "  new_keyframe_times_of_use: " << var_params_.new_keyframe_times_of_use << std::endl;
    std::cout << "  min_num_initial_map_kfs: " << min_num_initial_map_kfs_ << std::endl;
    std::cout << "  large_rot_th:            " << large_rot_th_ << std::endl;
    std::cout << "  large_trans_th:          " << large_trans_th_ << std::endl;
    std::cout << "  cull_keyframes:          " << cull_keyframes_ << std::endl;

    std::cout << "\n[CFG] Logging Parameters:" << std::endl;
    std::cout << "  training_report_interval: " << training_report_interval_ << std::endl;
    std::cout << "  keyframe_record_interval: " << keyframe_record_interval_ << std::endl;
    std::cout << "  all_keyframes_record_interval: " << all_keyframes_record_interval_ << std::endl;
    std::cout << "  record_rendered_image:   " << record_rendered_image_ << std::endl;
    std::cout << "  record_ground_truth_image: " << record_ground_truth_image_ << std::endl;
    std::cout << "  record_loss_image:       " << record_loss_image_ << std::endl;
}

void VoxelMapper::run() {
    py::gil_scoped_acquire gil;
    py::module_::import("sys").attr("path").attr("insert")(0, "../scripts_voxel");

    // ───── First loop: Initial voxel mapping ─────
    while (!isStopped()) {
        if (hasMetInitialMappingConditions()) {
            mpSLAM->getAtlas()->clearMappingOperation();

            // Step 1: extract initial keyframes and map points
            buildInitialKeyframesAndPointCloud();

            // Step 2: initialize voxel model
            initializeVoxelModel();

            // Step 3: first iteration
            trainForOneIteration();

            initial_mapped_ = true;
            break;
        } else if (mpSLAM->isShutDown()) {
            break;
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // ───── Second loop: Incremental voxel mapping ─────
    int slam_stop_iter = 0;
    while (!isStopped()) {
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_) cullKeyframes();
        }

        trainForOneIteration();

        if (mpSLAM->isShutDown() && !SLAM_ended_) {
            slam_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= mVoxelConfig.max_num_iterations)
            break;
    }

    // ───── Third loop: Tail voxel optimization ─────
    int densify_interval = mVoxelConfig.densification_interval_;
    int n_delay_iters = densify_interval * 0.8;

    while ((getIteration() - slam_stop_iter <= n_delay_iters) ||
           (getIteration() % densify_interval <= n_delay_iters) ||
           keep_training_)
    {
        trainForOneIteration();

        // Update adaptive delay
        densify_interval = mVoxelConfig.densification_interval_;
        n_delay_iters = densify_interval * 0.8;
    }

    // ───── Finalization ─────
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
    finalize();
    signalStop();
}

bool VoxelMapper::hasMetInitialMappingConditions() {
    return !mpSLAM->isShutDown() &&
           mpSLAM->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
           mpSLAM->getAtlas()->hasMappingOperation();
}

bool VoxelMapper::hasMetIncrementalMappingConditions() {
    return !mpSLAM->isShutDown() &&
           mpSLAM->getAtlas()->hasMappingOperation();
}

void VoxelMapper::buildInitialKeyframesAndPointCloud() {
    auto *pMap = mpSLAM->getAtlas()->GetCurrentMap();
    std::vector<ORB_SLAM3::KeyFrame*> vKFs;
    std::vector<ORB_SLAM3::MapPoint*> vMPs;

    {
        std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
        vKFs = pMap->GetAllKeyFrames();
        vMPs = pMap->GetAllMapPoints();
    }

    float voxel_size = 0.05f;
    std::set<std::vector<float>> unique_centres;
    for (auto *mp : vMPs) {
        if (!mp || mp->isBad()) continue;
        Eigen::Vector3f pos = mp->GetWorldPos().cast<float>();
        Eigen::Vector3f c   = (pos / voxel_size).array().round() * voxel_size;
        unique_centres.insert({c[0], c[1], c[2]});
    }

    std::vector<torch::Tensor> tensor_list;
    for (const auto& vec : unique_centres) {
        tensor_list.push_back(torch::from_blob((void*)vec.data(), {3}, torch::kFloat32).clone());
    }
    voxel_centers_ = torch::stack(tensor_list);

    for (auto *kf : vKFs) {
        if (!kf || kf->isBad()) continue;
        auto pkf = std::make_shared<VoxelKeyframe>();
        pkf->fid_ = kf->mnId;
        pkf->Tcw = Sophus::SE3f(kf->GetPose());
        pkf->img_path_ = (mSeqDir / "rgb" / (std::to_string(kf->mTimeStamp) + ".png")).string();
        pkf->remaining_times_of_use_ = 1;

        // std::cout << "[DEBUG] Initialized KF " << pkf->fid_
        //   << " with remaining uses = " << pkf->remaining_times_of_use_ << std::endl;
        // std::cout << "[INIT] Keyframe " << pkf->fid_
        //   << " initialized with uses = " << pkf->remaining_times_of_use_ << std::endl;
        // std::cout << "[CONFIG] new_keyframe_times_of_use = "
        //   << var_params_.new_keyframe_times_of_use << std::endl;
        mSceneKeyframes[pkf->fid_] = pkf;
        increaseKeyframeTimesOfUse(pkf, var_params_.new_keyframe_times_of_use);

        kfid_shuffled_ = false; 
    }
}

void VoxelMapper::initializeVoxelModel() {
    float voxel_size = 0.05f;
    const int64_t N = voxel_centers_.size(0);
    torch::Tensor oct_paths = torch::arange(N, torch::kLong).to(mDevice);
    torch::Tensor oct_levels = torch::zeros({N}, torch::kInt32).to(mDevice);
    torch::Tensor subdiv_meta = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(mDevice).requires_grad(true));
    torch::Tensor subdiv_p = torch::zeros_like(subdiv_meta, torch::kFloat32).to(mDevice).set_requires_grad(true);
    subdiv_meta.retain_grad();
    subdiv_p.retain_grad();  // ensure gradient is tracked

    mpTrainer->set_voxels(
        voxel_centers_.to(mDevice),
        torch::full({N}, voxel_size, torch::kFloat32).to(mDevice),
        torch::zeros({N,8}, torch::kFloat32).to(mDevice),
        torch::ones({N,3}, torch::kFloat32).to(mDevice) * 0.5f,
        torch::zeros({N,45}, torch::kFloat32).to(mDevice),
        torch::ones({N}, torch::kFloat32).to(mDevice) * 0.8f,
        oct_paths,
        oct_levels,
        subdiv_meta,
        subdiv_p
    );

    // optimizer_ = std::make_unique<torch::optim::Adam>(
    // mpTrainer->parameters(), torch::optim::AdamOptions(mVoxelConfig.lr));
    build_adam_optimizer(mVoxelConfig.lr);
    next_subdiv_iter_   = mVoxelConfig.subdiv_from;
    next_prune_iter_    = mVoxelConfig.prune_from;
    next_opacity_reset_ = mVoxelConfig.densification_interval_;
    std::cout << "[INFO] Voxel model initialized from SLAM map. "
              << "Adam groups = " << optimizer_->param_groups().size() << '\n';
}

void VoxelMapper::build_adam_optimizer(float base_lr)
{
    auto P = mpTrainer->parameters();   // 0:geo, 1:sh0, 2:opacity

    using torch::optim::Adam;
    using torch::optim::AdamOptions;
    using torch::optim::OptimizerParamGroup;

    AdamOptions opt(base_lr);  opt.eps(1e-15);
    optimizer_ = std::make_unique<Adam>(std::vector<torch::Tensor>{P[0]}, opt);   // geo_

    auto add = [&](torch::Tensor t, float lr)
    {
        OptimizerParamGroup g({t});
        optimizer_->add_param_group(g);
        static_cast<AdamOptions&>(optimizer_->param_groups().back().options())
            .lr(lr).eps(1e-15);
    };

    add(P[1], base_lr * 0.10f);   // sh0
    add(P[2], base_lr * 0.05f);   // opacity
    std::cout << "[OPT] Adam groups = " << optimizer_->param_groups().size() << '\n';
}

void VoxelMapper::generateKfidRandomShuffle()
{
    if (mSceneKeyframes.empty()) return;

    /* build deterministic list of KF IDs (keys of the unordered_map) */
    kfid_index_.clear();
    kfid_index_.reserve(mSceneKeyframes.size());
    for (auto const& kv : mSceneKeyframes)
        kfid_index_.push_back(kv.first);       // store the id only

    /* shuffle a vector of positions 0…N-1 */
    kfid_shuffle_.resize(kfid_index_.size());
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);

    std::mt19937 g{ rd_() };
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_    = true;
    kfid_shuffle_idx_ = -1;   // same convention as before
}

// std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomSlidingWindowKeyframe() {
//     if (mSceneKeyframes.empty()) return nullptr;
//     if (!kfid_shuffled_) generateKfidRandomShuffle();

//     int usable = 0;
//     for (const auto& [id, kf] : mSceneKeyframes) {
//         if (kf->remaining_times_of_use_ > 0) ++usable;
//     }
//     // std::cout << "[DEBUG] Keyframes with remaining_times_of_use_ > 0: " << usable << std::endl;

//     const int max_attempts = static_cast<int>(mSceneKeyframes.size()) * 2;
//     int attempts = 0;

//     while (attempts++ < max_attempts)
//     {
//         if (kfid_shuffle_idx_ >= static_cast<int>(kfid_shuffle_.size()))
//             kfid_shuffle_idx_ = 0;

//         int idx = kfid_shuffle_[kfid_shuffle_idx_++];
//         auto it = mSceneKeyframes.begin();
//         std::advance(it, idx);
//         auto viewpoint_cam = it->second;

//         if (viewpoint_cam->remaining_times_of_use_ > 0) {
//             --(viewpoint_cam->remaining_times_of_use_);
//             kfs_used_times_[viewpoint_cam->fid_]++;
//             return viewpoint_cam;
//         }
//     }

//     // All keyframes exhausted. Reset with fixed count!
//     std::cout << "[INFO] Reset keyframe usage counts.\n";
//     for (auto& kfit : mSceneKeyframes) {
//         kfit.second->remaining_times_of_use_ = 1;   // ← Set explicitly
//     }

//     kfid_shuffle_idx_ = 0;
//     return useOneRandomSlidingWindowKeyframe();
// }
std::shared_ptr<VoxelKeyframe> VoxelMapper::useOneRandomSlidingWindowKeyframe()
{
    if (mSceneKeyframes.empty()) return nullptr;
    if (!kfid_shuffled_) generateKfidRandomShuffle();

    const int nkfs = static_cast<int>(kfid_shuffle_.size());
    if (nkfs == 0) return nullptr;

    // --- Photo‑SLAM algorithm ----------------------------------------------------
    std::shared_ptr<VoxelKeyframe> viewpoint_cam = nullptr;

    const int start_shuffle_idx = kfid_shuffle_idx_;   // remember where we started
    do {
        // advance shuffled index (wrap‑around)
        kfid_shuffle_idx_ = (kfid_shuffle_idx_ + 1) % nkfs;

        // if we wrapped around ⇒ give *every* key‑frame one extra use
        if (kfid_shuffle_idx_ == start_shuffle_idx) {
            for (auto &kv : mSceneKeyframes)
                increaseKeyframeTimesOfUse(kv.second, 1); // +1 remaining_times_of_use_
        }

        // map shuffled position to actual key‑frame iterator
        int shuffled_pos = kfid_shuffle_[kfid_shuffle_idx_];
        unsigned long kfid = kfid_index_[shuffled_pos];
        viewpoint_cam = mSceneKeyframes.at(kfid);
        // auto it = mSceneKeyframes.begin();
        // std::advance(it, shuffled_pos);
        // viewpoint_cam = it->second;

    } while (viewpoint_cam->remaining_times_of_use_ <= 0);

    // ─── bookkeeping (exactly like GaussianMapper) ───
    const unsigned long fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(fid) == kfs_used_times_.end())
        kfs_used_times_[fid] = 1;
    else
        ++kfs_used_times_[fid];

    --(viewpoint_cam->remaining_times_of_use_);
    return viewpoint_cam;
}

void VoxelMapper::cullKeyframes()
{
    std::unordered_set<unsigned long> live_kfids =
        mpSLAM->getAtlas()->GetCurrentKeyFrameIds();

    std::vector<unsigned long> to_erase;
    to_erase.reserve(mSceneKeyframes.size());

    for (auto const& kv : mSceneKeyframes)
        if (live_kfids.find(kv.first) == live_kfids.end())
            to_erase.push_back(kv.first);

    for (auto id : to_erase)
        mSceneKeyframes.erase(id);

    if (!to_erase.empty())
        kfid_shuffled_ = false;       // ✱ add this
}

// ────────────────────────────────────────────────────────────────
//  Pull mapping operations from SLAM and update our scene
//  Currently we only react to LocalMappingBA by inserting /
//  updating key-frames.  Other operation types are ignored (safe).
// ────────────────────────────────────────────────────────────────
// ────────────────────────────────────────────────────────────────
//  Pull mapping operations from SLAM and update our scene
// ────────────────────────────────────────────────────────────────
void VoxelMapper::combineMappingOperations()
{
    while (mpSLAM->getAtlas()->hasMappingOperation())
    {
        ORB_SLAM3::MappingOperation opr =
            mpSLAM->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        // ——— LOCAL BA ————————————————————————————————
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            auto &kfs = opr.associatedKeyFrames();

            for (auto &kf_tuple : kfs)
            {
                unsigned long      kfid    = std::get<0>(kf_tuple);
                /* ------------------------------------------------------
                   Photo-SLAM puts the *original image filename* (string)
                   in std::get<1>(kf_tuple).  Using that guarantees we
                   never fall back to “…/rgb/0.png”.
                   ------------------------------------------------------ */
                const std::string &img_file = std::get<8>(kf_tuple);
                const Sophus::SE3f &Tcw_new = std::get<2>(kf_tuple);

                auto it = mSceneKeyframes.find(kfid);

                if (it != mSceneKeyframes.end())
                {   // already known → update pose only
                    it->second->Tcw = Tcw_new;
                }
                else
                {   // new key-frame → create & cache
                    auto pkf  = std::make_shared<VoxelKeyframe>();
                    pkf->fid_ = kfid;
                    pkf->Tcw  = Tcw_new;

                    /*  The filename coming from ORB-SLAM3 is already
                        correct (Photo-SLAM logic).  Prepend sequence
                        directory only if it is *relative*.            */
                    if (std::filesystem::path(img_file).is_absolute())
                        pkf->img_path_ = img_file;
                    else
                        pkf->img_path_ = (mSeqDir / img_file).string();

                    pkf->remaining_times_of_use_ = 0;
                    mSceneKeyframes[kfid] = pkf;

                    increaseKeyframeTimesOfUse(
                        pkf, var_params_.new_keyframe_times_of_use);
                }
            }
        }
        break;

        default: /* other operation types not needed for mapping yet */ break;
        }
    }
}

bool VoxelMapper::isStopped() const {
    return stopped_;
}

void VoxelMapper::signalStop(bool stop) {
    stopped_ = stop;
}

inline bool is_between(int iter, int from, int until)
{ return iter >= from && iter <= until; }

// Add training with optimization loop
void VoxelMapper::trainForOneIteration()
{   
    std::shared_ptr<VoxelKeyframe> pkf = useOneRandomSlidingWindowKeyframe();
    if (!pkf) {
        increaseIteration(-1);
        return;
    }
    // bookkeeping
    ++iteration_;

    /*–––––––––––– 2.   build MiniCam & ground-truth image  –––––––––––––––*/
    cv::Mat im = cv::imread(pkf->img_path_, cv::IMREAD_COLOR);
    if (im.empty()) return;
    cv::Mat imRGB;  cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);

    float fx = mpCamera->getParameter(0);
    float fy = mpCamera->getParameter(1);
    float cx = mpCamera->getParameter(2);
    float cy = mpCamera->getParameter(3);
    sv::MiniCam cam = sv::MiniCam::fromIntrinsics(
        fx, fy, cx, cy,
        im.cols, im.rows,
        static_cast<int>(pkf->fid_)
    );
    // Eigen::Matrix4f Tcw = mTcwList[i].matrix();
    Eigen::Matrix4f Tcw = pkf->Tcw.matrix();  // ← replace with your actual pose logic
    Eigen::Matrix4f c2w = Tcw.inverse();
    cam.c2w = torch::from_blob(c2w.data(), {4,4}, torch::kFloat32).clone().to(mDevice);
    cam.w2c = torch::from_blob(Tcw.data(), {4,4}, torch::kFloat32).clone().to(mDevice);

     /* gt tensor & mask (mask = valid-pixel region) */
    torch::Tensor gt = torch::from_blob(imRGB.data, {static_cast<int64_t>(cam.height), static_cast<int64_t>(cam.width), 3}, torch::kUInt8)
                            .permute({2, 0, 1})
                            .to(torch::kFloat32)
                            .div(255.0f)
                            .to(mDevice)
                            .clone();
    torch::Tensor mask = (gt.sum(0) > 0).to(torch::kFloat32);   // (H,W)

    /*–––––––––––– 3. update position LR as Photo-SLAM does ––––––––––––––*/
    int used_times = kfs_used_times_[pkf->fid_];
    int step       = std::min(used_times, pos_lr_max_steps_);
    float pos_lr   = mpTrainer->updateLearningRate(
                         step, lr_init_, pos_lr_max_steps_, pos_lr_delay_mult_);

    /* geo is param-group 0 */
    optimizer_->param_groups()[0].options().set_lr(pos_lr);

    /*–––––––––––– 4. render –––––––––––––––––––––––––––––––––––––––––––––*/
    py::array rgb_numpy = cvMatToNumpyRGB(imRGB);
    auto out_map  = mpTrainer->render(cam, rgb_numpy, "");
    torch::Tensor pred = out_map["rgb"].to(mDevice).squeeze(0);   // (3,H,W)

    TORCH_CHECK(pred.sizes() == gt.sizes(), "pred/gt size mismatch");

    /*–––––––––––– 5.   masked loss (L1 + dssim) ––––––––––––––––––––––––*/
    torch::Tensor p_masked = pred * mask;
    torch::Tensor g_masked = gt   * mask;

    auto Ll1  = loss_utils::l1_loss(p_masked, g_masked);
    float λ   = mVoxelConfig.lambda_ssim;                 // same as Gaussian
    auto p_contig = p_masked.contiguous();
    auto g_contig = g_masked.contiguous();
    auto dssim    = 1.0f - loss_utils::ssim(p_contig, g_contig, mDevice.type());

    torch::Tensor loss = (1.0f - λ) * Ll1 + λ * dssim;

    /*–––––––––––– 6. backward + optimiser step –––––––––––––––––––––––––*/
    optimizer_->zero_grad();
    {
        py::gil_scoped_release no_gil;  // release GIL for autograd
        loss.backward();
    }
    optimizer_->step();

    /*–––––––––––– 7. accumulate subdiv-priority gradient –––––––––––––––*/
    try {
        torch::NoGradGuard no_grad;

        torch::Tensor grad          = mpTrainer->get_subdiv_priority_grad();
        torch::Tensor grad_copy     = grad.clone();                 // safe
        torch::Tensor subdiv_meta   = mpTrainer->get_tensor("subdiv_meta");
        subdiv_meta += grad_copy * mVoxelConfig.meta_accum_lr;
        subdiv_meta = torch::where(torch::isfinite(subdiv_meta),
                                   subdiv_meta,
                                   torch::zeros_like(subdiv_meta))
                          .clamp(0.f, 1.f);

        mpTrainer->set_subdiv_meta(subdiv_meta);   // re-attach grad

        /* keep running buffer for later subdivision */
        torch::Tensor all_idx =
            torch::arange(grad.numel(), grad.options().dtype(torch::kLong));
        mpTrainer->accumulate_subdiv_gradients(all_idx, grad_copy);
    }
    catch (const std::exception& e) {
        std::cerr << "[WARN] subdivision-grad accumulation failed: "
                  << e.what() << std::endl;
    }

    /*–––––––––––– 8. exponential-moving-average loss for logging –––––––*/
    static float ema_loss = loss.item<float>();
    ema_loss = 0.4f * loss.item<float>() + 0.6f * ema_loss;

    if (iteration_ % 10 == 0) {
        std::cout << "[TRAIN] it "   << iteration_
                  << " | L1 "        << Ll1.item<float>()
                  << " | SSIM "      << (1.0f - dssim.item<float>())
                  << " | loss "      << loss.item<float>()
                //   << " | ema "       << ema_loss
                  << std::endl;
    }

    /*–––––––––––– 9. write usage table every 50 iters ––––––––––––––––––*/
    if (iteration_ % 50 == 0)
        writeKeyframeUsedTimes(result_dir_);
}

//------------------------------------------------------------
// helper: copy Mat → NumPy (NumPy owns the memory)
//------------------------------------------------------------
py::array_t<uint8_t> cvMatToNumpyRGB(const cv::Mat &img)
{
    if (img.empty() || img.type() != CV_8UC3)
        throw std::runtime_error("Expected a non-empty 3-channel CV_8UC3 image");

    // allocate new NumPy array that *owns* its data
    py::array_t<uint8_t> arr({img.rows, img.cols, 3});
    std::memcpy(arr.mutable_data(), img.data,
                static_cast<size_t>(img.rows * img.cols * 3));
    return arr;   // safe – Python holds the buffer
}

// void VoxelMapper::increaseKeyframeTimesOfUse(std::shared_ptr<VoxelKeyframe> pkf, int value)
// {
//     pkf->remaining_times_of_use_ += value;
//     // kfs_used_times_[pkf->fid_] = 0;  // reset usage tracking if needed
// }
inline void VoxelMapper::increaseKeyframeTimesOfUse(
        const std::shared_ptr<VoxelKeyframe>& kf,
        int n)
{
    if (!kf) return;
    kf->remaining_times_of_use_ += n;
    if (kf->remaining_times_of_use_ < 0)
        kf->remaining_times_of_use_ = 0;
}

void VoxelMapper::writeKeyframeUsedTimes(const std::filesystem::path& dir,
                                         const std::string& suffix)
{
    std::filesystem::create_directories(dir);
    std::ofstream f(dir / ("keyframe_used_times" + suffix + ".txt"),
                    std::ios::out | std::ios::app);

    f << "##[Voxel Mapper]Iteration " << getIteration()
      << " keyframe id, used times, remaining times:\n";

    for (auto const& kv : kfs_used_times_)
    {
        auto it = mSceneKeyframes.find(kv.first);
        int remaining = (it != mSceneKeyframes.end())
                        ? it->second->remaining_times_of_use_ : 0;
        f << kv.first << ' ' << kv.second << ' ' << remaining << '\n';
    }
    f << "##=========================================\n";
}

void VoxelMapper::renderAndRecordKeyframe(std::shared_ptr<VoxelKeyframe> pkf,
                                          float&  dssim,
                                          float&  psnr,
                                          double& render_ms,
                                          const std::filesystem::path& img_dir,
                                          const std::filesystem::path& gt_dir,
                                          const std::filesystem::path& loss_dir,
                                          const std::string& suffix)
{
    // ────── intrinsics from the ORB-SLAM3 camera ──────
    const float fx = mpCamera->getParameter(0);
    const float fy = mpCamera->getParameter(1);
    const float cx = mpCamera->getParameter(2);
    const float cy = mpCamera->getParameter(3);

    // ────── load the ground-truth frame (as RGB) ──────
    cv::Mat im_bgr = cv::imread(pkf->img_path_, cv::IMREAD_COLOR);
    if (im_bgr.empty()) return;

    cv::Mat im_rgb;
    cv::cvtColor(im_bgr, im_rgb, cv::COLOR_BGR2RGB);

    const int width  = im_rgb.cols;
    const int height = im_rgb.rows;

    /*  NumPy array that owns its memory — the Python renderer
        expects this.  */
    py::array rgb_numpy = cvMatToNumpyRGB(im_rgb);

    // ────── build MiniCam exactly like the Python side expects ──────
    const float tanfovx = 0.5f * width  / fx;   // tan(FOVx/2)
    const float tanfovy = 0.5f * height / fy;   // tan(FOVy/2)

    sv::MiniCam cam = sv::MiniCam::fromIntrinsics(
        fx, fy, cx, cy,
        width, height,
        static_cast<int>(pkf->fid_)
    );

    Eigen::Matrix4f Tcw = pkf->Tcw.matrix();
    Eigen::Matrix4f c2w = Tcw.inverse();

    cam.c2w = torch::from_blob(c2w.data(), {4,4}, torch::kFloat32).clone().to(mDevice);
    cam.w2c = torch::from_blob(Tcw.data(), {4,4}, torch::kFloat32).clone().to(mDevice);

    /* store the tangents so that the pybind converter can forward
       them to python-MiniCam */
    cam.fx      = fx;          // keep the usual intrinsics
    cam.fy      = fy;
    cam.cx      = cx;
    cam.cy      = cy;
    cam.width   = width;
    cam.height  = height;
    /* stash the tangents in the otherwise unused “near” & “frame_id”
       slots (cleaner than extending the struct here): */
    cam.near    = tanfovx;     // will be read in python as tanfovx
    cam.frame_id = *reinterpret_cast<const int*>(&tanfovy); // hacky but OK

    // ────── render ──────
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto out      = mpTrainer->render(cam, rgb_numpy, "");
    const auto t1 = std::chrono::high_resolution_clock::now();

    render_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    torch::Tensor rgb = out.at("rgb").cpu().squeeze(0);           // (3,H,W)

    // ────── optional dumping of the rendered frame ──────
    if (record_rendered_image_)
    {
        torch::Tensor img_u8 = (rgb.clamp(0,1) * 255)
                                .to(torch::kU8).permute({1,2,0}).contiguous();
        cv::Mat cvimg(img_u8.size(0), img_u8.size(1), CV_8UC3, img_u8.data_ptr());
        cv::cvtColor(cvimg, cvimg, cv::COLOR_RGB2BGR);

        cv::imwrite(img_dir / (std::to_string(getIteration()) + "_"
                               + std::to_string(pkf->fid_) + suffix + ".jpg"),
                    cvimg);
    }

    torch::Tensor pred = out.at("rgb").cpu().squeeze(0);  // (3,H,W)
    torch::Tensor gt_tensor = torch::from_blob(im_rgb.data, {height, width, 3}, torch::kU8)
                                .permute({2, 0, 1})
                                .to(torch::kFloat32)
                                .div(255.0f)
                                .clone();
    gt_tensor = gt_tensor.unsqueeze(0);  // (1,3,H,W)

    torch::Tensor pred_u = pred.unsqueeze(0);
    torch::Tensor gt_u   = gt_tensor;

    psnr  = loss_utils::psnr(pred_u, gt_u).item<float>();
    dssim = (1.0f - loss_utils::ssim(pred_u, gt_u)).item<float>();
}

void VoxelMapper::renderAndRecordAllKeyframes(const std::string& suffix)
{
    std::filesystem::path out_dir = result_dir_ /
                                    (std::to_string(getIteration()) + suffix);
    std::filesystem::create_directories(out_dir);
    std::filesystem::path img_dir  = out_dir / "image";
    std::filesystem::path gt_dir   = out_dir / "image_gt";
    std::filesystem::path loss_dir = out_dir / "image_loss";
    if (record_rendered_image_)        std::filesystem::create_directories(img_dir);
    if (record_ground_truth_image_)    std::filesystem::create_directories(gt_dir);
    if (record_loss_image_)            std::filesystem::create_directories(loss_dir);

    std::ofstream timings(out_dir / "render_time.txt");
    timings << "## keyframe id, time(ms)\n";

    std::ofstream psnr_f(out_dir / "psnr.txt");
    std::ofstream dssim_f(out_dir / "dssim.txt");
    psnr_f << "## kfid, PSNR\n";
    dssim_f << "## kfid, dssim\n";

    for (auto& kv : mSceneKeyframes) {
        float dssim, psnr;
        double ms;
        renderAndRecordKeyframe(kv.second, dssim, psnr, ms,
                                img_dir, gt_dir, loss_dir);
        timings << kv.first << ' ' << std::fixed << std::setprecision(6) << ms << '\n';
        psnr_f << kv.first << ' ' << psnr << '\n';
        dssim_f << kv.first << ' ' << dssim << '\n';
    }
}

/* --- Optional placeholders so the call-sites compile ------------------- */
void VoxelMapper::savePly(const std::filesystem::path&){ /* TODO when trainer supports export */ }
void VoxelMapper::keyframesToJson(const std::filesystem::path&){ }

void VoxelMapper::trainingReport(int iter,
                                 float loss,
                                 float ema_loss,
                                 double ms_per_iter,
                                 int kfid)
{
    std::cout << "[REPORT] iter " << iter
              << "  kf "  << kfid
              << "  loss " << std::fixed << std::setprecision(6) << loss
              << "  ema "  << ema_loss
              << "  "      << ms_per_iter << " ms\n";

    // dump to file (append)
    std::filesystem::path rpt_dir = mOutDir / (std::to_string(iter) + "_report");
    std::filesystem::create_directories(rpt_dir);
    std::ofstream fout(rpt_dir / "loss.txt", std::ios::app);
    fout << iter << " " << kfid << " " << loss << " " << ema_loss << "\n";
    fout.close();

    std::ofstream rt(rpt_dir / "render_time.txt", std::ios::app);
    rt << iter << " " << ms_per_iter << "\n";
}

/* ---------------- runtime getter / setter ---------------- */
VariableParameters VoxelMapper::getVariableParameters() const
{
    std::lock_guard<std::mutex> lk(mutex_status_);
    VariableParameters p;
    p.position_lr_init          = lr_init_;
    p.new_keyframe_times_of_use = var_params_.new_keyframe_times_of_use;
    p.do_inactive_geo_densify   = do_inactive_geo_densify_;
    p.keep_training = keep_training_;
    return p;
}

void VoxelMapper::setVariableParameters(const VariableParameters& p)
{
    std::lock_guard<std::mutex> lk(mutex_status_);
    /* apply only what VoxelMapper still honours */
    lr_init_                               = p.position_lr_init;
    var_params_.new_keyframe_times_of_use  = p.new_keyframe_times_of_use;
    do_inactive_geo_densify_               = p.do_inactive_geo_densify;
    keep_training_                         = p.keep_training;
}

cv::Mat VoxelMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));

    // Create MiniCam object with camera intrinsics
    float fx = mpCamera->getParameter(0);
    float fy = mpCamera->getParameter(1);
    float cx = mpCamera->getParameter(2);
    float cy = mpCamera->getParameter(3);

    const int w = (width  > 0 ? width  : 1);
    const int h = (height > 0 ? height : 1);
    
    sv::MiniCam cam = sv::MiniCam::fromIntrinsics(
        fx, fy, cx, cy,
        width, height,
        /*frame_id=*/0  // set to 0 or a dummy id
    );

    Eigen::Matrix4f Tcw_matrix = Tcw.matrix();
    Eigen::Matrix4f c2w_matrix = Tcw.inverse().matrix();

    cam.c2w = torch::from_blob(c2w_matrix.data(), {4,4}, torch::kFloat32).clone().to(mDevice);
    cam.w2c = torch::from_blob(Tcw_matrix.data(), {4,4}, torch::kFloat32).clone().to(mDevice);

    cv::Mat rendered_image;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);

        try {
            // Call voxel trainer’s render function
            auto out_map = mpTrainer->render(cam, py::none(), "viewer");

            torch::Tensor rendered_tensor = out_map["rgb"].cpu().squeeze(0);  // (3, H, W)
            rendered_tensor = rendered_tensor.permute({1, 2, 0});             // (H, W, 3)

            // Convert to OpenCV format
            cv::Mat image(height, width, CV_32FC3, rendered_tensor.data_ptr<float>());
            rendered_image = image.clone();  // ensure ownership
        }
        catch (const py::error_already_set& e) {          // <-- Python exceptions
            std::cerr << "[renderFromPose] Python exception:\n" 
                    << e.what() << std::endl;
            return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0,0,0));
        }
        catch (const std::exception& e) {
            std::cerr << "[VoxelMapper::renderFromPose] Rendering failed: " << e.what() << std::endl;
            return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
        }
    }
    return rendered_image;
}

void VoxelMapper::finalize()
{
    std::filesystem::create_directories(mOutDir);
    // auto model_path = mOutDir / "voxels.pth";
    // mpTrainer->save_torch(model_path);
    // Match GaussianMapper naming
    std::string tag = std::to_string(static_cast<int>(mImagePaths.size()));
    writeKeyframeUsedTimes(mOutDir, "_final");
    // Match trajectory/timing output of GaussianMapper
    const std::string out_dir_str = mOutDir.string();
    mpSLAM->SaveTrajectoryTUM(out_dir_str + "/CameraTrajectory_TUM.txt");
    mpSLAM->SaveKeyFrameTrajectoryTUM(out_dir_str + "/KeyFrameTrajectory_TUM.txt");
    mpSLAM->SaveTrajectoryEuRoC(out_dir_str + "/CameraTrajectory_EuRoC.txt");
    mpSLAM->SaveKeyFrameTrajectoryEuRoC(out_dir_str + "/KeyFrameTrajectory_EuRoC.txt");
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

std::filesystem::path VoxelMapper::getConfigFilePath() const {
    return config_file_path_;
}

bool VoxelMapper::isKeepingTraining() const {
    return keep_training_;
}

void VoxelMapper::setKeepTraining(bool v) {
    keep_training_ = v;
}

std::shared_ptr<sv::VoxelTrainer> VoxelMapper::getTrainer() const {
    return mpTrainer;
}