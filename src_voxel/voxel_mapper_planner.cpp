#include "include_voxel/voxel_mapper.h"
#include "include/stereo_vision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <opencv2/flann.hpp>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ORB-SLAM3/include/Atlas.h"
#include "ORB-SLAM3/include/MapPoint.h"

namespace py = pybind11;

namespace {
static inline torch::Tensor make_points_tensor_cpu_f32(
    const std::vector<Eigen::Vector3f>& pts
) {
    if (pts.empty()) {
        return torch::Tensor();
    }
    auto out = torch::empty(
        {(long)pts.size(), 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    );
    auto acc = out.accessor<float, 2>();
    for (int i = 0; i < (int)pts.size(); ++i) {
        acc[i][0] = pts[i].x();
        acc[i][1] = pts[i].y();
        acc[i][2] = pts[i].z();
    }
    return out;
}
static inline torch::Tensor make_single_point_cpu_f32(const Eigen::Vector3d& p) {
    return torch::tensor(
        {(float)p.x(), (float)p.y(), (float)p.z()},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
    ).view({1, 3});
}
static inline torch::Tensor color_u8(uint8_t r, uint8_t g, uint8_t b) {
    return torch::tensor({r, g, b}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
}
} // namespace

void VoxelMapper::runTsdfPlannerAtShutdown()
{
if (!planner_state_.planned_once_ && sensor_type_ == RGBD && sdf_params_.use_tsdf_planning_ && sdf_mapper_) {
        // Choose a stable iteration stamp for rerun. Best: last keyframe id or last training iter.
        // Here we use last keyframe id if possible; fallback to 0.
        int iteration0 = 0;
        if (scene_ && !scene_->keyframes().empty()) {
            iteration0 = (int)scene_->keyframes().rbegin()->first; // assumes ordered map
        }

        // 1) Update ESDF once (full layer) after mapping is complete
        sdf_mapper_->updateEsdf(nvblox::UpdateFullLayer::kYes);
        cudaDeviceSynchronize();

        auto printLayerAabb = [&](const auto& layer, const char* name) {
            const std::vector<nvblox::Index3D> blocks = layer.getAllBlockIndices();
            std::cout << "[NVBLOX] " << name << " blocks=" << blocks.size() << "\n";
            if (blocks.empty()) return;

            nvblox::Index3D mn = blocks[0], mx = blocks[0];
            for (const auto& b : blocks) {
                mn = mn.cwiseMin(b);
                mx = mx.cwiseMax(b);
            }

            const float vs = layer.voxel_size(); // meters
            const int vps = 8;                   // VoxelBlock::kVoxelsPerSide
            const float bs = vs * float(vps);    // block size in meters

            Eigen::Vector3f min_m(mn.x() * bs, mn.y() * bs, mn.z() * bs);
            Eigen::Vector3f max_m((mx.x() + 1) * bs, (mx.y() + 1) * bs, (mx.z() + 1) * bs);

            std::cout << "[NVBLOX] " << name << " AABB(m): ["
                    << min_m.transpose() << "] -> ["
                    << max_m.transpose() << "] voxel_size=" << vs << "\n";
        };

        printLayerAabb(sdf_mapper_->esdf_layer(), "ESDF");
        printLayerAabb(sdf_mapper_->tsdf_layer(), "TSDF");

        // 2) Bounds from ESDF blocks AABB (planner domain)
        PlanBounds b;
        {
            auto& layer = sdf_mapper_->esdf_layer();
            const std::vector<nvblox::Index3D> blocks = layer.getAllBlockIndices();
            TORCH_CHECK(!blocks.empty(), "ESDF has no blocks.");

            nvblox::Index3D mn = blocks[0], mx = blocks[0];
            for (const auto& bi : blocks) {
                mn = mn.cwiseMin(bi);
                mx = mx.cwiseMax(bi);
            }

            const float vs = layer.voxel_size();
            const double bs = double(vs) * 8.0; // 8 voxels/side
            const double pad = 0.5;             // meters padding around ESDF AABB

            b.min_x = mn.x() * bs - pad;
            b.min_y = mn.y() * bs - pad;
            b.min_z = mn.z() * bs - pad;

            b.max_x = (mx.x() + 1) * bs + pad;
            b.max_y = (mx.y() + 1) * bs + pad;
            b.max_z = (mx.z() + 1) * bs + pad;
        }

        // 3) Start pose from first keyframe
        Eigen::Vector3d start_pos;
        Eigen::Quaterniond start_q = Eigen::Quaterniond::Identity();
        {
            auto kf0_it = scene_->keyframes().begin();
            TORCH_CHECK(kf0_it != scene_->keyframes().end(), "No keyframes in scene");
            auto kf0 = kf0_it->second;
            TORCH_CHECK(kf0, "First keyframe is null");

            sv::MiniCam c0 = kf0->toMiniCam(kf0->image_height_, kf0->image_width_);
            torch::Tensor c2w = c0.c2w.to(torch::kCPU).contiguous(); // [4,4]
            TORCH_CHECK(c2w.sizes() == torch::IntArrayRef({4,4}), "MiniCam.c2w must be [4,4]");

            start_pos = Eigen::Vector3d(
                c2w[0][3].item<double>(),
                c2w[1][3].item<double>(),
                c2w[2][3].item<double>()
            );

            // Optional: orientation
            // Eigen::Matrix3d R;
            // for (int r=0;r<3;++r) for (int c=0;c<3;++c) R(r,c) = c2w[r][c].item<double>();
            // start_q = Eigen::Quaterniond(R);
        }

        // 4) Goal pose: last keyframe position (current choice)
        Eigen::Vector3d goal_pos;
        {
            auto& kfs = scene_->keyframes();
            TORCH_CHECK(!kfs.empty(), "No keyframes in scene");

            auto it = std::prev(kfs.end());
            auto kf_last = it->second;
            TORCH_CHECK(kf_last, "Last keyframe is null");

            sv::MiniCam cl = kf_last->toMiniCam(kf_last->image_height_, kf_last->image_width_);
            auto c2w = cl.c2w.to(torch::kCPU).contiguous();

            goal_pos = Eigen::Vector3d(
                c2w[0][3].item<double>(),
                c2w[1][3].item<double>(),
                c2w[2][3].item<double>()
            );
        }

        // 5) Instantiate planner ONCE with bounds + validity function
        if (!planner_state_.planner_) {
            planner_state_.planner_ = std::make_unique<VoxelPlanner>(b);

            const double robot_radius = static_cast<double>(planner_params_.planner_clearance_m_);
            planner_state_.planner_->setValidityFunction([this, robot_radius](const Eigen::Vector3d& p) -> bool {
                float d = 0.f;
                const bool ok = this->queryEsdfAtWorld(p, d);
                if (!ok) return false;                          // unknown => invalid (fail closed)
                return (static_cast<double>(d) >= robot_radius); // free if clearance satisfied
            });
        }

        auto in_bounds = [&](const Eigen::Vector3d& p){
            return (p.x() >= b.min_x && p.x() <= b.max_x &&
                    p.y() >= b.min_y && p.y() <= b.max_y &&
                    p.z() >= b.min_z && p.z() <= b.max_z);
        };

        std::cout << "[Planner] start: " << start_pos.transpose()
                << " in_bounds=" << in_bounds(start_pos) << "\n";
        std::cout << "[Planner] goal:  " << goal_pos.transpose()
                << " in_bounds=" << in_bounds(goal_pos) << "\n";
        std::cout << "[Planner] bounds: "
                << "["<<b.min_x<<","<<b.max_x<<"] "
                << "["<<b.min_y<<","<<b.max_y<<"] "
                << "["<<b.min_z<<","<<b.max_z<<"]\n";

        // ---------------------------------------------------------
        // A) Rerun: log start & goal markers
        // ---------------------------------------------------------
        {
            torch::Tensor start_t = make_single_point_cpu_f32(start_pos);
            torch::Tensor goal_t  = make_single_point_cpu_f32(goal_pos);

            // Color them explicitly so it's always obvious.
            torch::Tensor green = torch::tensor({0,255,0}, torch::TensorOptions().dtype(torch::kUInt8)).view({1,3});
            torch::Tensor red   = torch::tensor({255,0,0}, torch::TensorOptions().dtype(torch::kUInt8)).view({1,3});

            sv::RerunVisualizerBridge::instance().visualizePoints3D(
                start_t, green, iteration0, "world/planner/start", 0.08f);

            sv::RerunVisualizerBridge::instance().visualizePoints3D(
                goal_t, red, iteration0, "world/planner/goal", 0.08f);
        }

        // --- ESDF sanity at start/goal ---
        auto dbg_esdf = [&](const char* tag, const Eigen::Vector3d& p) {
            float d = 0.f;
            const bool ok = this->queryEsdfAtWorld(p, d);
            std::cout << "[Planner] ESDF " << tag
                    << " ok=" << ok
                    << " d=" << d
                    << " clearance=" << planner_params_.planner_clearance_m_
                    << "\n";
        };
        dbg_esdf("start", start_pos);
        dbg_esdf("goal",  goal_pos);

        // ---------------------------------------------------------
        // B) Rerun: log "ESDF-known free samples" (key visual)
        // ---------------------------------------------------------
        {
            const int N = 25000; // dense cloud for a good "voxblox-like" free-space feeling
            std::vector<Eigen::Vector3f> free_pts;
            free_pts.reserve(N);

            std::mt19937 rng(0);
            std::uniform_real_distribution<double> ux(b.min_x, b.max_x);
            std::uniform_real_distribution<double> uy(b.min_y, b.max_y);
            std::uniform_real_distribution<double> uz(b.min_z, b.max_z);

            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d p(ux(rng), uy(rng), uz(rng));
                float d = 0.f;
                const bool ok = this->queryEsdfAtWorld(p, d);
                if (!ok) continue;                        // ESDF unknown -> ignore
                if (d < planner_params_.planner_clearance_m_) continue;    // not free enough

                free_pts.emplace_back(p.cast<float>());
            }

            std::cout << "[Planner] free_samples kept=" << free_pts.size() << " / " << N << "\n";

            torch::Tensor pts = make_points_tensor_cpu_f32(free_pts);
            if (pts.defined() && pts.numel() > 0) {
                // Light gray points to mimic "free space" feel.
                torch::Tensor gray = torch::full({(long)free_pts.size(), 3}, 180,
                    torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

                sv::RerunVisualizerBridge::instance().visualizePoints3D(
                    pts, gray, iteration0, "world/esdf/free_samples", 0.01f);
            }
        }

        // Optional: quick stats (keep if you like)
        {
            const int N = 3000;
            int ok_cnt = 0;
            int valid_cnt = 0;

            std::mt19937 rng(0);
            std::uniform_real_distribution<double> ux(b.min_x, b.max_x);
            std::uniform_real_distribution<double> uy(b.min_y, b.max_y);
            std::uniform_real_distribution<double> uz(b.min_z, b.max_z);

            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d p(ux(rng), uy(rng), uz(rng));
                float d = 0.f;
                const bool ok = this->queryEsdfAtWorld(p, d);
                ok_cnt += ok ? 1 : 0;
                if (!ok) continue;
                if (d >= planner_params_.planner_clearance_m_) valid_cnt++;
            }

            std::cout << "[Planner] samples N=" << N
                    << " esdf_ok=" << ok_cnt << " (" << (ok_cnt/(double)N) << ")"
                    << " valid=" << valid_cnt << " (" << (valid_cnt/(double)N) << ")"
                    << "\n";
        }

        // 6) Plan
        const double solve_seconds = 10.0;
        PlanResult res = planner_state_.planner_->plan(start_pos, start_q, goal_pos, solve_seconds);

        if (res.success) {
            const auto out_path = (result_dir_ / "planned_path_xyz.txt").string();
            std::ofstream f(out_path);
            for (const auto& p : res.waypoints) {
                f << p.x() << " " << p.y() << " " << p.z() << "\n";
            }
            std::cout << "[Planner] success. waypoints=" << res.waypoints.size()
                    << " length=" << res.length << "\n";

            // ---------------------------------------------------------
            // A) Rerun: log planned path (static)
            // ---------------------------------------------------------
            if (res.waypoints.size() >= 2) {
                std::vector<Eigen::Vector3f> path_pts;
                path_pts.reserve(res.waypoints.size());
                for (const auto& p : res.waypoints) {
                    path_pts.emplace_back((float)p.x(), (float)p.y(), (float)p.z());
                }

                torch::Tensor path_t = make_points_tensor_cpu_f32(path_pts);

                // Path color: green
                torch::Tensor green = color_u8(0, 255, 0);

                sv::RerunVisualizerBridge::instance().visualizeLineStrip3D(
                    path_t, green, iteration0, "world/planner/path", 0.02f);

                // ---------------------------------------------------------
                // Robot playback along the path (SplatNav-like)
                // We log "world/planner/robot" at successive iter steps.
                // ---------------------------------------------------------
                {
                    // robot marker color: blue
                    torch::Tensor blue = torch::tensor({0, 120, 255}, torch::TensorOptions().dtype(torch::kUInt8))
                                            .view({1,3});

                    // How many frames to animate:
                    // If you want smoother motion, resample between waypoints later.
                    int base_iter = iteration0 + 1;

                    for (int i = 0; i < (int)path_pts.size(); ++i) {
                        Eigen::Vector3d p(path_pts[i].x(), path_pts[i].y(), path_pts[i].z());
                        torch::Tensor robot_t = make_single_point_cpu_f32(p);

                        sv::RerunVisualizerBridge::instance().visualizePoints3D(
                            robot_t, blue, base_iter + i, "world/planner/robot", (float)planner_params_.planner_clearance_m_);
                    }
                }
            }

        } else {
            std::cout << "[Planner] failed.\n";

            // Optional: log straight line in red to show "why planning matters".
            // (Leave commented if you prefer clean output.)
            /*
            std::vector<Eigen::Vector3f> seg;
            seg.emplace_back((float)start_pos.x(), (float)start_pos.y(), (float)start_pos.z());
            seg.emplace_back((float)goal_pos.x(),  (float)goal_pos.y(),  (float)goal_pos.z());
            torch::Tensor seg_t = make_points_tensor_cpu_f32(seg);
            torch::Tensor red = color_u8(255,0,0);
            sv::RerunVisualizerBridge::instance().visualizeLineStrip3D(
                seg_t, red, iteration0, "world/planner/failed_straight_line", 0.02f);
            */
        }

        planner_state_.planned_once_ = true;
    }
}

bool VoxelMapper::queryEsdfAtWorld(const Eigen::Vector3d& p_W, float& dist_out) const
{
    if (!sdf_mapper_) return false;

    nvblox::EsdfLayer& esdf_layer = sdf_mapper_->esdf_layer();
    if (esdf_layer.size() == 0) return false;

    std::vector<nvblox::Vector3f> positions;
    positions.emplace_back(
        static_cast<float>(p_W.x()),
        static_cast<float>(p_W.y()),
        static_cast<float>(p_W.z())
    );

    std::vector<nvblox::EsdfVoxel> voxels;
    std::vector<bool> success;
    esdf_layer.getVoxels(positions, &voxels, &success);

    if (success.empty() || !success[0]) return false;

    const nvblox::EsdfVoxel& v = voxels[0];

    // If voxel never observed, treat as unknown.
    if (!v.observed) return false;

    const float voxel_size = sdf_mapper_->voxel_size_m();  // Mapper::voxel_size_m()
    const float dist_m = std::sqrt(std::max(0.0f, v.squared_distance_vox)) * voxel_size;

    dist_out = v.is_inside ? -dist_m : dist_m;
    return true;
}
