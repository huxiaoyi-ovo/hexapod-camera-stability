// Deterministic command-line adapter around the unmodified OpenSHC gait core.
// One STEP request advances WalkController exactly once at the configured 20 ms.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <ros/ros.h>

#include "syropod_highlevel_controller/debug_visualiser.h"
#include "syropod_highlevel_controller/model.h"
#include "syropod_highlevel_controller/pose_controller.h"
#include "syropod_highlevel_controller/walk_controller.h"

namespace {
const std::vector<std::string> kLegs{"rf", "rm", "rb", "lf", "lm", "lb"};
const std::vector<std::string> kJoints{"thigh", "knee", "ankle"};

template <typename T>
void set_parameter(Parameter<T>& parameter, const T& value) {
  parameter.data = value;
  parameter.initialised = true;
}

void set_adjustable(AdjustableParameter& parameter, double value) {
  parameter.data = {{"default", value}, {"min", value}, {"max", value}, {"step", 0.0}};
  parameter.current_value = parameter.default_value = value;
  parameter.min_value = parameter.max_value = value;
  parameter.adjust_step = 0.0;
  parameter.initialised = true;
}

struct LegInit {
  std::string id;
  Eigen::Vector3d mount_xyz;
  Eigen::Vector3d mount_rpy;
  Eigen::Vector3d initial_q;
  Eigen::Vector3d initial_foot_body;
  double plane_to_body_toe_z = 0.0;
  std::array<std::array<double, 4>, 3> limits{};
  std::array<std::array<double, 4>, 4> dh{};
};

struct InitData {
  double dt = 0.0;
  double native_frequency = 0.0;
  int stance_phase = 0;
  int swing_phase = 0;
  int phase_offset = 0;
  double swing_height = 0.0;
  double speed_limit = 0.0;
  double angular_limit = 0.0;
  double acceleration_limit = 0.0;
  std::array<LegInit, 6> legs{};
};

bool read_init(InitData& data) {
  std::string tag;
  if (!(std::cin >> tag) || tag != "INIT" ||
      !(std::cin >> data.dt >> data.native_frequency >> data.stance_phase >> data.swing_phase >>
        data.phase_offset >> data.swing_height >> data.speed_limit >> data.angular_limit >>
        data.acceleration_limit)) {
    return false;
  }
  for (std::size_t i = 0; i < kLegs.size(); ++i) {
    LegInit& leg = data.legs[i];
    if (!(std::cin >> leg.id) || leg.id != kLegs[i]) return false;
    for (int j = 0; j < 3; ++j) if (!(std::cin >> leg.mount_xyz[j])) return false;
    for (int j = 0; j < 3; ++j) if (!(std::cin >> leg.mount_rpy[j])) return false;
    for (int j = 0; j < 3; ++j) if (!(std::cin >> leg.initial_q[j])) return false;
    for (int j = 0; j < 3; ++j) if (!(std::cin >> leg.initial_foot_body[j])) return false;
    if (!(std::cin >> leg.plane_to_body_toe_z)) return false;
    for (auto& row : leg.limits) for (double& value : row) if (!(std::cin >> value)) return false;
    for (auto& row : leg.dh) for (double& value : row) if (!(std::cin >> value)) return false;
  }
  return data.dt > 0.0 && data.native_frequency > 0.0 && data.swing_height > 0.0 &&
         data.speed_limit > 0.0 && data.angular_limit > 0.0 && data.acceleration_limit > 0.0 &&
         data.stance_phase > 0 && data.swing_phase > 0;
}

bool finite_vector(const Eigen::Vector3d& value) {
  return value.array().isFinite().all();
}

std::shared_ptr<Model> make_model(const InitData& init, Parameters& params) {
  set_parameter(params.time_delta, init.dt);
  set_parameter(params.leg_id, kLegs);
  set_parameter(params.joint_id, kJoints);
  set_parameter(params.link_id, std::vector<std::string>{"root", "thigh", "knee", "toe"});
  std::map<std::string, int> dof;
  for (const auto& leg : kLegs) dof[leg] = 3;
  set_parameter(params.leg_DOF, dof);

  set_parameter(params.clamp_joint_positions, true);
  set_parameter(params.clamp_joint_velocities, true);
  set_parameter(params.ignore_IK_warnings, false);
  set_parameter(params.debug_IK, false);
  set_parameter(params.debug_rviz, false);
  set_parameter(params.debug_workspace_calc, false);
  set_parameter(params.debug_swing_trajectory, false);
  set_parameter(params.debug_stance_trajectory, false);
  set_parameter(params.rough_terrain_mode, false);
  set_parameter(params.gravity_aligned_tips, false);
  set_parameter(params.force_normal_touchdown, false);
  set_parameter(params.admittance_control, false);
  set_parameter(params.overlapping_walkspaces, false);
  set_parameter(params.velocity_input_mode, std::string("real"));
  set_parameter(params.leg_manipulation_mode, std::string("tip_control"));
  double mean_body_to_toe = 0.0;
  for (const auto& leg : init.legs) mean_body_to_toe -= leg.initial_foot_body.z();
  set_parameter(params.body_clearance, mean_body_to_toe / init.legs.size());
  set_parameter(params.max_translation_velocity, init.speed_limit);
  set_parameter(params.max_rotation_velocity, init.angular_limit);
  set_parameter(params.touchdown_threshold, 1.0);
  set_parameter(params.liftoff_threshold, 0.5);
  set_adjustable(params.step_frequency, init.native_frequency);
  set_adjustable(params.swing_height, init.swing_height);
  set_adjustable(params.swing_width, 0.0);
  set_adjustable(params.step_depth, 0.0);
  set_adjustable(params.stance_span_modifier, 0.0);
  set_adjustable(params.force_gain, 1.0);
  set_parameter(params.stance_phase, init.stance_phase);
  set_parameter(params.swing_phase, init.swing_phase);
  set_parameter(params.phase_offset, init.phase_offset);
  std::map<std::string, int> offsets{{"rf", 1}, {"rm", 0}, {"rb", 1},
                                     {"lf", 0}, {"lm", 1}, {"lb", 0}};
  set_parameter(params.offset_multiplier, offsets);

  for (std::size_t i = 0; i < init.legs.size(); ++i) {
    const LegInit& leg = init.legs[i];
    params.leg_stance_positions[i].data = {{"x", leg.initial_foot_body.x()},
                                           {"y", leg.initial_foot_body.y()}};
    params.leg_stance_positions[i].initialised = true;
    for (int j = 0; j < 3; ++j) {
      params.joint_parameters[i][j].data = {{"min", leg.limits[j][0]}, {"max", leg.limits[j][1]},
                                            {"offset", 0.0}, {"unpacked", leg.initial_q[j]},
                                            {"max_vel", leg.limits[j][3]}, {"packed", 0.0}};
      params.joint_parameters[i][j].initialised = true;
    }
    for (int j = 0; j < 4; ++j) {
      params.link_parameters[i][j].data = {{"r", leg.dh[j][0]}, {"theta", leg.dh[j][1]},
                                           {"d", leg.dh[j][2]}, {"alpha", leg.dh[j][3]}};
      params.link_parameters[i][j].initialised = true;
    }
  }

  auto model = std::allocate_shared<Model>(Eigen::aligned_allocator<Model>(), params,
                                           std::shared_ptr<DebugVisualiser>());
  const Pose body_above_walk_plane(Eigen::Vector3d(0.0, 0.0, mean_body_to_toe / init.legs.size()),
                                   Eigen::Quaterniond::Identity());
  model->setCurrentPose(body_above_walk_plane);
  model->setDefaultPose(body_above_walk_plane);
  model->generate();
  model->initLegs(true);
  for (const auto& init_leg : init.legs) {
    auto leg = model->getLegByIDName(init_leg.id);
    if (!leg) throw std::runtime_error("OpenSHC leg name mismatch: " + init_leg.id);
    for (int j = 0; j < 3; ++j) {
      auto joint = leg->getJointByIDName(init_leg.id + "_" + kJoints[j] + "_joint");
      if (!joint) throw std::runtime_error("OpenSHC joint name mismatch");
      joint->current_position_ = init_leg.initial_q[j];
      joint->desired_position_ = init_leg.initial_q[j];
    }
    leg->applyFK(true, false);
  }
  return model;
}

bool run_protocol(InitData& init) {
  Parameters params{};
  std::shared_ptr<Model> model = make_model(init, params);
  auto walker = std::allocate_shared<WalkController>(Eigen::aligned_allocator<WalkController>(), model, params);
  walker->init();
  std::map<std::string, int> pose_phase_by_leg;
  std::map<std::string, double> zero_pose_by_leg;
  for (const auto& leg : kLegs) {
    pose_phase_by_leg[leg] = 0;
    zero_pose_by_leg[leg] = 0.0;
  }
  set_parameter(params.pose_frequency, -1.0);
  set_parameter(params.pose_phase_length, 4);
  set_parameter(params.pose_phase_starts, std::vector<int>{0});
  set_parameter(params.pose_phase_ends, std::vector<int>{0});
  set_parameter(params.pose_negation_phase_starts, pose_phase_by_leg);
  set_parameter(params.pose_negation_phase_ends, pose_phase_by_leg);
  set_parameter(params.negation_transition_ratio, zero_pose_by_leg);
  set_parameter(params.x_amplitudes, std::vector<double>{0.0});
  set_parameter(params.y_amplitudes, std::vector<double>{0.0});
  set_parameter(params.z_amplitudes, std::vector<double>{0.0});
  set_parameter(params.gravity_amplitudes, std::vector<double>{0.0});
  set_parameter(params.roll_amplitudes, std::vector<double>{0.0});
  set_parameter(params.pitch_amplitudes, std::vector<double>{0.0});
  set_parameter(params.yaw_amplitudes, std::vector<double>{0.0});
  auto poser = std::allocate_shared<PoseController>(Eigen::aligned_allocator<PoseController>(), model, params);
  poser->init();
  model->updateDefaultConfiguration();
  model->generateWorkspaces();
  walker->setPoseState(POSING_COMPLETE);
  walker->generateWalkspace();
  LimitMap forward_speed_limits;
  walker->generateLimits(&forward_speed_limits, nullptr, nullptr, nullptr);
  const double forward_speed_limit = walker->getLimit(Eigen::Vector2d(0.0, 1.0), 0.0,
                                                       forward_speed_limits);
  const StepCycle cycle = walker->getStepCycle();
  double max_fk_error = 0.0;
  std::array<double, 6> fk_errors{};
  for (std::size_t i = 0; i < init.legs.size(); ++i) {
    const auto& init_leg = init.legs[i];
    auto leg = model->getLegByIDName(init_leg.id);
    fk_errors[i] = (leg->getCurrentTipPose().position_ - init_leg.initial_foot_body).norm();
    max_fk_error = std::max(max_fk_error, fk_errors[i]);
  }
  std::cout << std::setprecision(17) << "READY " << cycle.period_ << " " << cycle.frequency_ << " "
            << max_fk_error << " " << forward_speed_limit << "\n";
  for (std::size_t i = 0; i < init.legs.size(); ++i)
    std::cout << "FK " << init.legs[i].id << " " << fk_errors[i] << "\n";
  std::cout << "END\n" << std::flush;

  std::size_t tick = 0;
  std::string tag;
  while (std::cin >> tag) {
    if (tag == "QUIT") return true;
    if (tag != "STEP") return false;
    double requested_x = 0.0, requested_y = 0.0, requested_omega = 0.0;
    if (!(std::cin >> requested_x >> requested_y >> requested_omega)) return false;
    std::array<std::array<double, 3>, 6> measured_q{};
    for (auto& leg_q : measured_q) for (double& q : leg_q) if (!(std::cin >> q) || !std::isfinite(q)) return false;

    for (std::size_t i = 0; i < init.legs.size(); ++i) {
      auto leg = model->getLegByIDName(init.legs[i].id);
      for (int j = 0; j < 3; ++j) {
        auto joint = leg->getJointByIDName(init.legs[i].id + "_" + kJoints[j] + "_joint");
        joint->current_position_ = measured_q[i][j];
        joint->desired_position_ = measured_q[i][j];
      }
      leg->applyFK(true, false);
    }

    walker->updateWalk(Eigen::Vector2d(requested_x, requested_y), requested_omega);
    const Eigen::Vector2d accepted = walker->getDesiredLinearVelocity();
    const double accepted_omega = walker->getDesiredAngularVelocity();
    const Pose odometry = walker->getOdometryIdeal();
    const StepCycle active_cycle = walker->getStepCycle();
    ++tick;
    std::cout << std::setprecision(17) << "OK " << tick << " " << requested_x << " " << requested_y << " "
              << requested_omega << " " << accepted.x() << " " << accepted.y() << " " << accepted_omega << " "
              << odometry.position_.x() << " " << odometry.position_.y() << " " << odometry.position_.z() << " "
              << odometry.rotation_.x() << " " << odometry.rotation_.y() << " " << odometry.rotation_.z() << " "
              << odometry.rotation_.w() << " " << active_cycle.period_ << "\n";

    for (std::size_t i = 0; i < init.legs.size(); ++i) {
      const auto& init_leg = init.legs[i];
      auto leg = model->getLegByIDName(init_leg.id);
      auto stepper = leg->getLegStepper();
      const Pose target = stepper->getCurrentTipPose();
      Pose model_target = Pose::Undefined();
      // The stepper target is already body-centred in the walk-plane frame.
      // Feed that target directly to the leg model; only transform it to
      // odometry coordinates for the external world-position trace.
      model_target.position_ = target.position_;
      model_target.position_.z() += init_leg.plane_to_body_toe_z;
      leg->setDesiredTipPose(model_target, false);
      const double ik_quality = leg->applyIK(false);
      const double residual = (leg->getCurrentTipPose().position_ - model_target.position_).norm();
      const Eigen::Vector3d target_odom = odometry.transformVector(target.position_);
      std::array<double, 3> q_target{}, dq_target{};
      std::array<int, 3> velocity_limit_active{}, position_limit_active{};
      for (int j = 0; j < 3; ++j) {
        auto joint = leg->getJointByIDName(init_leg.id + "_" + kJoints[j] + "_joint");
        q_target[j] = joint->desired_position_;
        dq_target[j] = joint->desired_velocity_;
        velocity_limit_active[j] = std::abs(dq_target[j]) >= init_leg.limits[j][3] - 1e-7;
        position_limit_active[j] = std::abs(q_target[j] - init_leg.limits[j][0]) < 1e-7 ||
                                   std::abs(q_target[j] - init_leg.limits[j][1]) < 1e-7;
      }
      const bool support = stepper->getStepState() == STANCE || stepper->getStepState() == FORCE_STANCE;
      std::cout << "LEG " << init_leg.id << " " << stepper->getPhase() << " " << stepper->getStepState() << " "
                << static_cast<double>(stepper->getPhase()) / active_cycle.period_ << " "
                << stepper->getStanceProgress() << " " << stepper->getSwingProgress() << " "
                << target_odom.x() << " " << target_odom.y() << " " << target_odom.z() << " "
                << q_target[0] << " " << q_target[1] << " " << q_target[2] << " "
                << dq_target[0] << " " << dq_target[1] << " " << dq_target[2] << " "
                << velocity_limit_active[0] << " " << velocity_limit_active[1] << " " << velocity_limit_active[2] << " "
                << position_limit_active[0] << " " << position_limit_active[1] << " " << position_limit_active[2] << " "
                << static_cast<int>(support) << " " << residual << " " << ik_quality << "\n";
    }
    std::cout << "END\n" << std::flush;
  }
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "openshc_bridge", ros::init_options::NoSigintHandler | ros::init_options::NoRosout);
  ros::Time::init();
  InitData init;
  try {
    if (!read_init(init)) {
      std::cerr << "expected one valid INIT request on stdin\n";
      return 2;
    }
    if (!run_protocol(init)) {
      std::cerr << "invalid OpenSHC line protocol request\n";
      return 3;
    }
  } catch (const std::exception& error) {
    std::cerr << "OpenSHC bridge error: " << error.what() << "\n";
    return 4;
  }
  return 0;
}
