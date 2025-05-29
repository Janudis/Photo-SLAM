// Add training with optimization loop
void VoxelMapper::trainForOneIteration()
{
    // std::cout << "[INFO] Iter " << iteration_
    //           << " | Total available keyframes: " << mSceneKeyframes.size() << std::endl;
    // auto optimizer = std::make_unique<torch::optim::Adam>(
    //     mpTrainer->parameters(), torch::optim::AdamOptions(mVoxelConfig.lr));
    
    // Extract loss weights from config
    float w_photo    = mVoxelConfig.lambda_photo;
    float w_ssim     = mVoxelConfig.lambda_ssim;
    float w_tconcen  = mVoxelConfig.lambda_T_concen;
    float w_tinside  = mVoxelConfig.lambda_T_inside;

    bool changed = false;  // tracks if subdivide/prune occurred
    auto iter_start = std::chrono::steady_clock::now();
    int total_kfs = static_cast<int>(mSceneKeyframes.size());
    // std::cout << "[INFO] Iter " << iter << " | Total available keyframes: " << total_kfs << std::endl;
    float total_loss = 0.f;
    int frames_used = 0;

    // const int iter = iteration_;
    // std::shared_ptr<VoxelKeyframe> pkf = useOneRandomSlidingWindowKeyframe();
    // if (!pkf) {
    //     std::cout << "[INFO] No valid keyframe left to use. Ending iteration early.\n";
    //     return;  
    // }
    // --(pkf->remaining_times_of_use_);
    // // std::cout << "[INFO] Using keyframe index " << pkf->fid_
    // //           << " (path = " << pkf->img_path_ << ")\n";
    // // std::cout << "[INFO] Remaining uses for keyframe "
    // //           << pkf->fid_ << ": "
    // //           << pkf->remaining_times_of_use_ << std::endl;
    // ++iteration_;
    // // kfs_used_times_[pkf->fid_]++;

    const int iter = iteration_;
    ++iteration_;
    auto pkf = useOneRandomSlidingWindowKeyframe();
    if (!pkf) return;
    /* position LR schedule */
    int used = kfs_used_times_[pkf->fid_];
    float pos_lr = mpTrainer->updateLearningRate(
                    std::min(used, pos_lr_max_steps_),
                    lr_init_, pos_lr_max_steps_, pos_lr_delay_mult_);
    optimizer_->param_groups()[0].options().set_lr(pos_lr);

    cv::Mat im = cv::imread(pkf->img_path_, cv::IMREAD_COLOR);
    if (im.empty()) return;
    cv::Mat imRGB;
    cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);
    py::array rgb_numpy = cvMatToNumpyRGB(imRGB);

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

    torch::Tensor gt = torch::from_blob(imRGB.data, {static_cast<int64_t>(cam.height), static_cast<int64_t>(cam.width), 3}, torch::kUInt8)
                            .permute({2, 0, 1})
                            .to(torch::kFloat32)
                            .div(255.0f)
                            .to(mDevice)
                            .clone();
    try {
        auto subdiv_p = mpTrainer->get_tensor("subdiv_p");

        {
            int used_times = kfs_used_times_[pkf->fid_];
            int step       = std::min(used_times, pos_lr_max_steps_);
            float lr       = mpTrainer->updateLearningRate(
                                    step,
                                    lr_init_,
                                    pos_lr_max_steps_,
                                    pos_lr_delay_mult_);

            // std::cout << "[DEBUG] Position learning rate this iter: " << lr << std::endl;
            // Re-build Adam if LR changed noticeably (optional):
            // (*optimizer).param_groups()[0].options().set_lr(lr);
            // for (auto& group : optimizer_->param_groups()) {
            //     group.options().set_lr(lr);
            // }
        }
        auto out_map = mpTrainer->render(cam, rgb_numpy, "");
        torch::Tensor pred = out_map["rgb"].to(mDevice);
        // torch::Tensor gt_tensor = out_map["gt"].to(mDevice);  // shape (1,3,H,W)
        torch::Tensor gt_tensor = gt.unsqueeze(0);   // (1,3,H,W)
        
        TORCH_CHECK(pred.sizes() == gt_tensor.sizes(), "pred and gt_tensor shapes must match");
        // drop the interpreter lock for the heavy autograd work
        py::gil_scoped_release no_gil;

        // torch::Tensor l2   = torch::mse_loss(pred, gt_tensor);
        // torch::Tensor loss = w_photo * l2;
        // if (out_map.find("T") != out_map.end())
        // {
        //     torch::Tensor T         = out_map["T"].to(mDevice);   // (1,1,H,W)
        //     torch::Tensor T_concen  = (T * (1.0f - T)).mean();
        //     torch::Tensor T_inside  = T.square().mean();
        //     loss += w_tconcen * T_concen + w_tinside * T_inside;
        // }
        // SSIM term
        // auto ssim_loss   = 1.0f - loss_utils::ssim(pred_contig, gt_contig, mDevice.type());
        // loss += w_ssim * ssim_loss;

        auto pred_contig = pred.contiguous();
        auto gt_contig   = gt_tensor.contiguous();
        auto Ll1 = loss_utils::l1_loss(pred_contig, gt_contig);
        float lambda_dssim = w_ssim;
        auto ssim_term = 1.0f - loss_utils::ssim(pred_contig, gt_contig, mDevice.type());
        auto loss = (1.0f - lambda_dssim) * Ll1 + lambda_dssim * ssim_term;

        // if (iteration_ % 10 == 0)   // print every 10 iterations (tweak as you like)
        // {
        //     std::cout << "[DBG-LOSS] iter " << iteration_
        //             << " | L2   = " << l2.item<float>()
        //             << " (weight " << w_photo << ')'
        //             << " | SSIM = " << ssim_loss.item<float>()
        //             << " (weight " <<  w_ssim << ')';
        //     std::cout << "[DBG-LR] adam lr = "
        //       << (*optimizer).param_groups()[0].options().get_lr()
        //       << '\n';

        //     if (out_map.find("T") != out_map.end())
        //     {
        //         std::cout << " | T_concen = "
        //                 << (out_map["T"] * (1.0f - out_map["T"])).mean().item<float>()
        //                 << " (w " << w_tconcen << ')'
        //                 << " | T_inside = "
        //                 << out_map["T"].square().mean().item<float>()
        //                 << " (w " << w_tinside << ')';
        //     }
        //     std::cout << " | TOTAL = " << loss.item<float>() << '\n';
        // }

        // Backward pass
        // (*optimizer).zero_grad();
        optimizer_->zero_grad();
        loss.backward();
        // std::cout << "[DEBUG] Loss value: " << loss.item<float>() << std::endl;
        // std::cout << "[DEBUG] SSIM Score: " << (1.0f - ssim_loss.item<float>()) << std::endl;

        // torch::Tensor geo_dbg = mpTrainer->get_tensor("geo");   // <-- fetch
        // std::cerr << "[DBG]  geo.requires_grad = "   << geo_dbg.requires_grad()
        //         << ", is_leaf = "                  << geo_dbg.is_leaf()
        //         << ", grad_defined = "
        //         << (geo_dbg.grad().defined() ? "yes" : "no") << '\n';
        // if (geo_dbg.grad().defined())
        //     std::cerr << "[DBG]  geo.grad  min/max = "
        //             << geo_dbg.grad().min().item<float>() << " / "
        //             << geo_dbg.grad().max().item<float>() << '\n';

        auto subdiv_grad = mpTrainer->get_subdiv_priority_grad();
        // if (subdiv_p.grad().defined()) {
        //     std::cout << "[DEBUG] subdiv_p.grad().max(): " << subdiv_p.grad().max().item<float>() << std::endl;
        // } else {
        //     std::cout << "[DEBUG] subdiv_p.grad() is NOT defined!" << std::endl;
        // }
        // std::cout << "==== Per-tensor grad norms after backward ====" << std::endl;
        // auto tensors = mpTrainer->parameters();
        // std::vector<std::string> names = {"center_", "size_", "geo_", "sh0_", "shs_", "opacity_", "subdiv_meta_"};
        // for (size_t i = 0; i < tensors.size(); ++i) {
        //     if (tensors[i].grad().defined()) {
        //         std::cout << "[DEBUG] " << names[i]
        //                 << " grad norm: "
        //                 << tensors[i].grad().norm().item<float>() << std::endl;
        //     } else {
        //         std::cout << "[DEBUG] " << names[i]
        //                 << " grad not defined!" << std::endl;
        //     }
        // }
        // std::cout << "==============================================" << std::endl;
        // std::cout << "==== Per-optimizer param check ====" << std::endl;
        // for (const auto& p : mpTrainer->parameters()) {
        //     std::cout << "[DEBUG] Param: shape=" << p.sizes()
        //             << ", requires_grad=" << p.requires_grad()
        //             << ", is_leaf=" << p.is_leaf()
        //             << ", grad_defined=" << (p.grad().defined() ? "yes" : "no")
        //             << std::endl;
        // }

        // (*optimizer).step();                // ← Applies optimizer step
        optimizer_->step();
        // std::cout << "[DEBUG] Optimizer stepped." << std::endl;
        // std::cout << "[DEBUG] sh0_.mean(): " << mpTrainer->get_tensor("sh0").mean().item<float>() << std::endl;
        // std::cout << "[DEBUG] opacity_.mean(): " << mpTrainer->get_tensor("opacities").mean().item<float>() << std::endl;

        // === Accumulate gradient for subdivision ===
        try {
            torch::NoGradGuard no_grad;
            auto grad = mpTrainer->get_subdiv_priority_grad();   // ← Safe accessor
            auto grad_copy = grad.clone();
            // grab current subdiv_meta (still a leaf)
            auto subdiv_meta = mpTrainer->get_tensor("subdiv_meta");
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
        std::cout << "[CPP-DBG] Python raised → we skipped this key-frame\n" << std::endl;
        return;
    }
    // }
    
    // if (mVoxelConfig.subdiv_every > 0 && (iteration_ + 1) % mVoxelConfig.subdiv_every == 0) 
    // {
    //     if (iter >= mVoxelConfig.subdiv_from && iter <= mVoxelConfig.subdiv_until) {
    //         torch::Tensor grad;
    //         try {
    //             grad = mpTrainer->get_subdiv_priority_grad();  // ← uses subdiv_p_.grad()
    //         } catch (const std::exception& e) {
    //             std::cerr << "[WARN] Cannot get subdivision gradients: " << e.what() << "\n";
    //             return;
    //         }

    //         if (!grad.defined() || grad.numel() == 0) return;

    //         torch::Tensor grad_detached = grad.detach();

    //         if (!grad_detached.isfinite().all().item<bool>()) {
    //             std::cerr << "[WARN] subdiv_p.grad contains NaNs or infs. Skipping...\n";
    //             return;
    //         }

    //         float thresh = grad_detached.quantile(mVoxelConfig.subdiv_quantile).item<float>();

    //         auto size_ten = mpTrainer->get_tensor("size");
    //         torch::Tensor can_split = (size_ten * 0.5) >= sv::MIN_VOX_SIZE;

    //         torch::Tensor valid_mask = mpTrainer->get_tensor("oct_level") < sv::MAX_VOXEL_LEVEL;

    //         torch::Tensor mask = (grad_detached > thresh)
    //                         & (grad_detached > mVoxelConfig.subdiv_gradient_threshold)
    //                         & can_split
    //                         & valid_mask;

    //         int64_t n_sub = mask.sum().item<int64_t>();

    //         if (n_sub > 0) {
    //             std::cout << "[SUBDIV] iter " << iter << "  |  splitting " << n_sub << " voxels\n";
    //             mpTrainer->subdivide(mask);  // also resets subdiv_p_ internally
    //             auto new_meta = torch::zeros_like(mpTrainer->get_tensor("subdiv_p"));  // now subdiv_p has correct shape
    //             mpTrainer->set_subdiv_meta(new_meta);  // this will call .retain_grad() as needed

    //             changed = true;
    //             std::cout << "[INFO] Voxels after subdivide: " << mpTrainer->num_voxels() << "\n";
    //         }
    //     }
    // }
    // // === Pruning ===
    // int64_t before = 0;
    // int64_t after  = 0;
    // torch::Tensor keep_mask;

    // if ((iter >= mVoxelConfig.prune_from) &&
    //     (iter <= mVoxelConfig.prune_until) &&
    //     (mVoxelConfig.prune_every > 0) &&
    //     (iter % mVoxelConfig.prune_every == 0)) {

    //     torch::NoGradGuard _;

    //     auto size_ten  = mpTrainer->get_tensor("size");
    //     auto meta_ten  = mpTrainer->get_tensor("subdiv_meta");

    //     // Adaptive prune threshold
    //     float prune_iter_rate = float(iter - mVoxelConfig.prune_from) /
    //                             float(mVoxelConfig.prune_until - mVoxelConfig.prune_from);
    //     prune_iter_rate = std::clamp(prune_iter_rate, 0.f, 1.f);

    //     float thresh = mVoxelConfig.prune_threshold_init +
    //                 (mVoxelConfig.prune_threshold_final - mVoxelConfig.prune_threshold_init) * prune_iter_rate;

    //     // 1. Remove invalid values (NaNs or infs)
    //     meta_ten = torch::where(torch::isfinite(meta_ten), meta_ten, torch::zeros_like(meta_ten));
    //     // 2. Compute adaptive threshold via quantile (SVRaster-style)
    //     float quant = meta_ten.quantile(mVoxelConfig.subdiv_quantile).item<float>();
    //     // 3. Create mask: keep voxels with meta >= quantile threshold
    //     torch::Tensor keep_mask = (meta_ten >= quant) & (size_ten >= sv::MIN_VOX_SIZE);

    //     if (keep_mask.sum().item<int64_t>() < mVoxelConfig.min_voxels)
    //         keep_mask.slice(0, 0, mVoxelConfig.min_voxels).fill_(true);

    //     int64_t before = mpTrainer->num_voxels();
    //     int64_t after  = keep_mask.sum().item<int64_t>();

    //     if (after < before) {
    //         mpTrainer->prune(keep_mask);
    //         std::cout << "[PRUNE] " << before << " → " << after << " voxels\n";
    //         changed = true;
    //     }

    //     std::cout << "[DEBUG] After pruning: total_loss = " << total_loss
    //             << " | frames_used = " << frames_used
    //             << " | average = " << (frames_used > 0 ? total_loss / frames_used : 0.0f)
    //             << std::endl;

    //     meta_ten.detach_().zero_();  // reset stats
    // }
    // if (changed) {
    //     // optimizer = build_optimizer();   // grab fresh leaf tensors
    //     optimizer = std::make_unique<torch::optim::Adam>(
    //         mpTrainer->parameters(), torch::optim::AdamOptions(mVoxelConfig.lr));
    //     std::cout << "[INFO] Rebuilt optimizer after topology change\n";
    //     changed = false;
    // }   

    // std::cout << "[KF USAGE] Iter " << iteration_
    //       << " | Used KF " << pkf->fid_
    //       << " | Remaining = " << pkf->remaining_times_of_use_
    //       << " | Total used = " << kfs_used_times_[pkf->fid_]
    //       << std::endl;

    if (iteration_ % 50 == 0) {
        writeKeyframeUsedTimes(result_dir_);
    }

    std::cout << "[TRAIN] Iter " << iteration_
            << " | Used " << frames_used << " keyframe out of "
            << mSceneKeyframes.size()
            << " | Avg Loss: " << (frames_used ? total_loss / frames_used : 0.f)
            << std::endl;
    // std::cout << "[TRAIN] Iter " << iter
    //     << " | Used " << frames_used << " keyframes out of " << total_kfs
    //     << " | Avg Loss: " << (frames_used > 0 ? total_loss / frames_used : 0.0)
    //     << std::endl;
}
// }