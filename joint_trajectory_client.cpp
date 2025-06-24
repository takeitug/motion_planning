#include <chrono>
#include <memory>
#include <string>
#include <thread>  // 追加：sleepのため

// include fri for number of joints
#include "friLBRState.h"

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

// SwitchController サービス用
#include "controller_manager_msgs/srv/switch_controller.hpp"
#include "controller_manager_msgs/srv/load_controller.hpp"
#include "controller_manager_msgs/srv/configure_controller.hpp"

#define PI 3.14159265358979323846
double rad2deg(double radians) {
    return radians * (180.0 / PI);
}
double deg2rad(double degrees) {
    return degrees * (PI / 180.0);
}

class JointTrajectoryClient : public rclcpp::Node {
public:
  JointTrajectoryClient(const std::string &node_name) : Node(node_name) {
    joint_trajectory_action_client_ =
        rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(
            this, "joint_trajectory_controller/follow_joint_trajectory");

    switch_controller_client_ =
        this->create_client<controller_manager_msgs::srv::SwitchController>(
            "/lbr/controller_manager/switch_controller");

    load_controller_client_ =
        this->create_client<controller_manager_msgs::srv::LoadController>(
            "/lbr/controller_manager/load_controller");

    configure_controller_client_ =
        this->create_client<controller_manager_msgs::srv::ConfigureController>(
            "/lbr/controller_manager/configure_controller");

    while (!joint_trajectory_action_client_->wait_for_action_server(std::chrono::seconds(1))) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(this->get_logger(),
                     "Interrupted while waiting for the action server. Exiting.");
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Waiting for action server to become available...");
    }
    RCLCPP_INFO(this->get_logger(), "Action server available.");
  };

  void execute(const std::vector<double> &positions, const int32_t &sec_from_start = 10) {
    if (positions.size() != KUKA::FRI::LBRState::NUMBER_OF_JOINTS) {
      RCLCPP_ERROR(this->get_logger(), "Invalid number of joint positions.");
      return;
    }

    control_msgs::action::FollowJointTrajectory::Goal joint_trajectory_goal;
    int32_t goal_sec_tolerance = 1;
    joint_trajectory_goal.goal_time_tolerance.sec = goal_sec_tolerance;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = positions;
    point.velocities.resize(KUKA::FRI::LBRState::NUMBER_OF_JOINTS, 0.0);
    point.time_from_start.sec = sec_from_start;

    for (std::size_t i = 0; i < KUKA::FRI::LBRState::NUMBER_OF_JOINTS; ++i) {
      joint_trajectory_goal.trajectory.joint_names.push_back("lbr_A" + std::to_string(i + 1));
    }

    joint_trajectory_goal.trajectory.points.push_back(point);

    // send goal
    auto goal_future = joint_trajectory_action_client_->async_send_goal(joint_trajectory_goal);
    rclcpp::spin_until_future_complete(this->get_node_base_interface(), goal_future);
    auto goal_handle = goal_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server.");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Goal was accepted by server.");

    // wait for result
    auto result_future = joint_trajectory_action_client_->async_get_result(goal_handle);
    rclcpp::spin_until_future_complete(this->get_node_base_interface(), result_future,
                                       std::chrono::seconds(sec_from_start + goal_sec_tolerance));
    if (result_future.get().result->error_code !=
        control_msgs::action::FollowJointTrajectory::Result::SUCCESSFUL) {
      RCLCPP_ERROR(this->get_logger(), "Failed to execute joint trajectory.");
      return;
    }
  }

  void switch_to_twist_controller() {
    load_controller_if_needed("twist_controller");
    configure_controller("twist_controller");
    while (!switch_controller_client_->wait_for_service(std::chrono::seconds(1))) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for service. Exiting.");
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Waiting for controller_manager service...");
    }

    auto request = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
    request->activate_controllers.push_back("twist_controller");
    request->deactivate_controllers.push_back("joint_trajectory_controller");
    request->strictness = 2;  // STRICT モードで確実に切り替える

    auto future = switch_controller_client_->async_send_request(request);
    rclcpp::spin_until_future_complete(this->get_node_base_interface(), future);

    if (future.get()->ok) {
      RCLCPP_INFO(this->get_logger(), "Successfully switched to twist_controller.");
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to switch controllers.");
    }
  }

  void switch_to_joint_controller() {
    while (!switch_controller_client_->wait_for_service(std::chrono::seconds(1))) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for service. Exiting.");
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Waiting for controller_manager service...");
    }

    auto request = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
    request->activate_controllers.push_back("joint_trajectory_controller");
    request->deactivate_controllers.push_back("twist_controller");
    request->strictness = 2;  // STRICT モードで確実に切り替える

    auto future = switch_controller_client_->async_send_request(request);
    rclcpp::spin_until_future_complete(this->get_node_base_interface(), future);

    if (future.get()->ok) {
      RCLCPP_INFO(this->get_logger(), "Successfully switched to joint_controller.");
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to switch controllers.");
    }
  }

  bool load_controller_if_needed(const std::string &controller_name) {
    if (!load_controller_client_) {
      RCLCPP_ERROR(this->get_logger(), "load_controller_client_ is not initialized.");
      return false;
    }
  
    while (!load_controller_client_->wait_for_service(std::chrono::seconds(1))) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for load_controller service.");
        return false;
      }
      RCLCPP_INFO(this->get_logger(), "Waiting for load_controller service...");
    }
  
    auto request = std::make_shared<controller_manager_msgs::srv::LoadController::Request>();
    request->name = controller_name;
  
    auto future = load_controller_client_->async_send_request(request);
    rclcpp::spin_until_future_complete(this->get_node_base_interface(), future);
  
    auto response = future.get();
    if (response && response->ok) {
      RCLCPP_INFO(this->get_logger(), "Controller '%s' loaded successfully.", controller_name.c_str());
      return true;
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to load controller '%s'.", controller_name.c_str());
      return false;
    }
  }

  bool configure_controller(const std::string &name) {
    auto request = std::make_shared<controller_manager_msgs::srv::ConfigureController::Request>();
    request->name = name;
  
    auto future = configure_controller_client_->async_send_request(request);
    rclcpp::spin_until_future_complete(this->get_node_base_interface(), future);
  
    auto response = future.get();
    if (response && response->ok) {
      RCLCPP_INFO(this->get_logger(), "Successfully configured controller '%s'.", name.c_str());
      return true;
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to configure controller '%s'.", name.c_str());
      return false;
    }
  }

protected:
  rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::SharedPtr joint_trajectory_action_client_;
  rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr switch_controller_client_;
  rclcpp::Client<controller_manager_msgs::srv::LoadController>::SharedPtr load_controller_client_;
  rclcpp::Client<controller_manager_msgs::srv::ConfigureController>::SharedPtr configure_controller_client_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto joint_trajectory_client = std::make_shared<JointTrajectoryClient>("joint_trajectory_client");

  // 1回目の動作
  RCLCPP_INFO(joint_trajectory_client->get_logger(), "Rotating odd joints.");
  joint_trajectory_client->execute({
      deg2rad(-20.0),
      deg2rad(80.0),
      deg2rad(100.0),
      deg2rad(-80.0),
      deg2rad(-80.0),
      deg2rad(-80.0),
      deg2rad(20.0),
  });

// コントローラーを twist_controller に切り替え
  
  RCLCPP_INFO(joint_trajectory_client->get_logger(), "Switching to twist_controller.");
  joint_trajectory_client->switch_to_twist_controller();

  joint_trajectory_client->switch_to_joint_controller();

  rclcpp::shutdown();
  return 0;
}