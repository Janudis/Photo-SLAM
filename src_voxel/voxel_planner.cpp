#include "voxel_planner.h"
#include <stdexcept>

VoxelPlanner::VoxelPlanner(const PlanBounds& bounds) : bounds_(bounds) {
  setupOMPL();
}

void VoxelPlanner::setValidityFunction(ValidityFn fn) {
  validity_fn_ = std::move(fn);
}

void VoxelPlanner::setupOMPL() {
  auto space = std::make_shared<ob::SE3StateSpace>();
  ob::RealVectorBounds b(3);

  b.setLow(0, bounds_.min_x); b.setHigh(0, bounds_.max_x);
  b.setLow(1, bounds_.min_y); b.setHigh(1, bounds_.max_y);
  b.setLow(2, bounds_.min_z); b.setHigh(2, bounds_.max_z);

  space->setBounds(b);

  setup_ = std::make_shared<og::SimpleSetup>(space);
  setup_->setStateValidityChecker([this](const ob::State* s) { return isStateValid(s); });

  setup_->getSpaceInformation()->setup();

  ob::OptimizationObjectivePtr obj(
      new ob::PathLengthOptimizationObjective(setup_->getSpaceInformation()));
  setup_->setOptimizationObjective(obj);

  auto planner = std::make_shared<og::RRTstar>(setup_->getSpaceInformation());
  // Optional but often helpful tuning (safe defaults):
  planner->setGoalBias(0.05);      // 5% goal bias
  planner->setRange(0.5);          // step size (meters) — tune to your scene scale

  setup_->setPlanner(planner);

  setup_->setup();
}

bool VoxelPlanner::isStateValid(const ob::State* state) {
  if (!validity_fn_) {
    // If no validity function is installed, fail closed.
    return false;
  }
  const auto* s = state->as<ob::SE3StateSpace::StateType>();
  Eigen::Vector3d p(s->getX(), s->getY(), s->getZ());
  return validity_fn_(p);
}

PlanResult VoxelPlanner::plan(const Eigen::Vector3d& start_pos,
                              const Eigen::Quaterniond& start_q,
                              const Eigen::Vector3d& goal_pos,
                              double solve_seconds) {
  if (!setup_) throw std::runtime_error("OMPL setup not initialized.");

  setup_->clear();

  auto si = setup_->getSpaceInformation();
  ob::ScopedState<ob::SE3StateSpace> start(si);
  ob::ScopedState<ob::SE3StateSpace> goal(si);

  start->setXYZ(start_pos.x(), start_pos.y(), start_pos.z());
  start->rotation().x = start_q.x();
  start->rotation().y = start_q.y();
  start->rotation().z = start_q.z();
  start->rotation().w = start_q.w();

  goal->setXYZ(goal_pos.x(), goal_pos.y(), goal_pos.z());
  goal->rotation().setIdentity();

  const double goal_tolerance = 0.3; // meters, tune
  setup_->setStartAndGoalStates(start, goal, goal_tolerance);

  const double straight_cost = (goal_pos - start_pos).norm();
  setup_->getOptimizationObjective()->setCostThreshold(ob::Cost(2.0 * straight_cost));

  std::cout << "[Planner] ||start-goal|| = " << (goal_pos - start_pos).norm() << "\n";
  std::cout << "[Planner] goal_tolerance = " << goal_tolerance << "\n";

  const ob::PlannerStatus solved = setup_->solve(solve_seconds);

  PlanResult out;
  if (solved == ob::PlannerStatus::StatusType::EXACT_SOLUTION) {
    setup_->simplifySolution();
    auto path = setup_->getSolutionPath();
    path.interpolate();

    out.success = true;
    out.length = path.length();
    out.waypoints.reserve(path.getStateCount());

    for (std::size_t i = 0; i < path.getStateCount(); ++i) {
      const auto* st = path.getState(i)->as<ob::SE3StateSpace::StateType>();
      out.waypoints.emplace_back(st->getX(), st->getY(), st->getZ());
    }
  }
  return out;
}
