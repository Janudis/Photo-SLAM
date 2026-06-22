#pragma once

#include <Eigen/Core>

#include <memory>

#include "include_voxel/voxel_planner.h"

struct VoxelPlannerParameters
{
    Eigen::Vector3f offline_goal_W_ = Eigen::Vector3f(10, 5, 0);
    float planner_clearance_m_ = 0.3f;
};

struct VoxelPlannerState
{
    std::unique_ptr<VoxelPlanner> planner_;
    bool planned_once_ = false;
};
