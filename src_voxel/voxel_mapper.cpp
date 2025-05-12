#include "include_voxel/voxel_mapper.h"
#include "include_voxel/voxel_trainer.h"
#include "include_voxel/mini_cam.h"
#include "include_voxel/py_utils.h"

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

namespace py = pybind11;

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
            const std::filesystem::path& voxel_cfg,
            const std::filesystem::path& seq_dir,
            const std::filesystem::path& out_dir,
            torch::Device device)
    :   mpSLAM(slam),
        mVoxelCfg(voxel_cfg),
        mSeqDir(seq_dir),
        mOutDir(out_dir),
        mDevice(device)
{
    // py::initialize_interpreter();

    // std::fprintf(stderr, "[DBG]  main interp  = %p\n", PyThreadState_Get()->interp);

    // py::module_ sys = py::module_::import("sys");
    // sys.attr("path").attr("insert")(0, "../scripts_voxel");

    // /* Drop the GIL *for good* – store the thread‑state so we can restore it */
    // m_savedState = PyEval_SaveThread();      // releases the lock

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
    /*   Create + own a complete interpreter in THIS thread */
    // py::scoped_interpreter guard{};
    // make our helper modules visible
    py::module_::import("sys")
        .attr("path").attr("insert")(0, "../scripts_voxel");

    // std::fprintf(stderr, "[DBG]  current interp = %p\n", PyThreadState_Get()->interp);

    LoadImages((mSeqDir / "rgb.txt").string());
 
    // Phase 1: Feed all images to SLAM
    size_t kf_index = 0;          // <- new
    for (size_t i = 0; i < mImagePaths.size(); ++i) 
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
    // std::cout << "[DEBUG] Tracking state = " << mpSLAM->GetTrackingState() << "\n";

    // Phase 2: Check if SLAM initialized properly
    int n_kfs = mpSLAM->GetNumKeyframes();
    // std::cout << "[DEBUG] Final keyframes: " << n_kfs << "\n";
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
    // std::cout << "[DEBUG] Tracked keyframe IDs: " << mKeyframeIds.size() << "\n";

    // Phase 3: Initialize voxels from SLAM map
    float voxel_size = 0.05f;
    torch::Tensor voxel_centers;
    if (!initializeVoxelsFromMap(voxel_centers, voxel_size)) {
        std::cout << "[ERROR] No valid map points after SLAM.\n";
        return;
    }

    const int64_t N = voxel_centers.size(0);
    /*  simple but sufficient: give each voxel a unique 64‑bit id
        0,1,2,…  (Morton codes can come later)                                   */
    torch::Tensor oct_paths = torch::arange(N, torch::kLong).to(mDevice);

    mpTrainer->set_voxels(
            voxel_centers.to(mDevice),                           // (N,3)
            torch::full({N},   voxel_size, torch::kFloat32).to(mDevice),      // lengths
            torch::zeros({N,8},  torch::kFloat32).to(mDevice),   // geo key‑pts (unused)
            torch::ones ({N,3},  torch::kFloat32).to(mDevice)*0.5f,           // RGB
            torch::zeros({N,45}, torch::kFloat32).to(mDevice),   // SH coeffs
            torch::ones ({N},    torch::kFloat32).to(mDevice)*0.8f,           // opacity
            oct_paths                                             /*  <‑‑  */
    );
    std::cout << "[INFO] Voxel model initialized from SLAM map.\n";

    // Phase 4: Train on all frames again with optimization
    trainLoopWithOptimization(50);  // or any number of iterations you prefer
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
        auto out_map = mpTrainer->render(cam, rgb_numpy, img_dir.string());
        at::Tensor rgb_tensor_f32 = out_map.at("rgb").cpu();                 // (3,H,W) float32
        at::Tensor rgb_tensor_u8  = (rgb_tensor_f32.clamp(0,1) * 255.0f)
                                        .to(torch::kU8)
                                        .permute({1,2,0})        // → (H,W,3) for OpenCV
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

// Add training with optimization loop
void VoxelMapper::trainLoopWithOptimization(int num_iters)
{
    torch::optim::Adam optimizer(mpTrainer->parameters(), torch::optim::AdamOptions(1e-2));
    
    for (int iter = 0; iter < num_iters; ++iter)
    {
        float total_loss = 0.f;
        int loss_count = 0;
        int frames_used = 0;

        for (size_t i = 0; i < mImagePaths.size(); ++i)
        {
            if (mTrackState[i] != 1) continue;

            cv::Mat im = cv::imread(mImagePaths[i], cv::IMREAD_COLOR);
            if (im.empty()) continue;

            cv::Mat imRGB;
            cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);
            py::array rgb_numpy = cvMatToNumpyRGB(imRGB);

            sv::MiniCam cam;
            cam.width = im.cols; cam.height = im.rows;
            cam.cx = cam.width * 0.5f; cam.cy = cam.height * 0.5f;
            cam.fovx = 2.f * std::atan(cam.width  / (2.f * 517.306f));
            cam.fovy = 2.f * std::atan(cam.height / (2.f * 516.469f));
            cam.frame_id = static_cast<int>(i);

            Eigen::Matrix4f Tcw = mTcwList[i].matrix();
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
                auto out_map = mpTrainer->render(cam, rgb_numpy, "");
                torch::Tensor pred = out_map["rgb"].to(mDevice);
                
                if (pred.sizes() != gt.sizes()) continue;

                 // drop the interpreter lock for the heavy autograd work
                py::gil_scoped_release no_gil;
                optimizer.zero_grad();
                torch::Tensor loss = torch::mse_loss(pred, gt);
                loss.backward();
                optimizer.step();
                total_loss += loss.item<float>();
                // GIL automatically re‑acquired here
                frames_used++;

            } catch (const py::error_already_set& e) {
                std::cerr << "[Python-Error] " << e.what() << std::endl;
                continue;
            }
        }
        std::cout << "[TRAIN] Iter " << iter << "  |  Loss: "
                  << (frames_used > 0 ? total_loss / frames_used : 0.0) << std::endl;
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

