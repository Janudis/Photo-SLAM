#pragma once

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <functional>

#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>

namespace ob = ompl::base;
namespace og = ompl::geometric;

// Forward declare your mapper/model types (include real headers in .cpp)
namespace sv { class VoxelModel; }

struct PlanBounds {
  double min_x, max_x;
  double min_y, max_y;
  double min_z, max_z;
};

struct PlanResult {
  bool success = false;
  std::vector<Eigen::Vector3d> waypoints;
  double length = 0.0;
};

class VoxelPlanner {
public:
  VoxelPlanner(const PlanBounds& bounds);

  // Provide access to whatever collision oracle you want:
  // easiest: a callback that answers "isFree(p)?".
  using ValidityFn = std::function<bool(const Eigen::Vector3d&)>;

  void setValidityFunction(ValidityFn fn);

  // Offline plan
  PlanResult plan(const Eigen::Vector3d& start_pos,
                  const Eigen::Quaterniond& start_q,
                  const Eigen::Vector3d& goal_pos,
                  double solve_seconds = 5.0);

private:
  void setupOMPL();
  bool isStateValid(const ob::State* state);

  PlanBounds bounds_;
  ValidityFn validity_fn_;

  std::shared_ptr<og::SimpleSetup> setup_;
};
