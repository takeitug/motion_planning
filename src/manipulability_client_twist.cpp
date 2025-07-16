#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <algorithm>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include "eigen3/Eigen/Dense"

#include "motion_planning/inverse.h"
#include "motion_planning/manipulability_gradient.h"
#include "motion_planning/forward.h"

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "controller_manager_msgs/srv/switch_controller.hpp"
#include "controller_manager_msgs/srv/load_controller.hpp"
#include "controller_manager_msgs/srv/configure_controller.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <limits>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include "motion_planning/nanoflann.hpp"

#define PI 3.14159265358979323846
double rad2deg(double radians) {
    return radians * (180.0 / PI);
}
double deg2rad(double degrees) {
    return degrees * (PI / 180.0);
}

//robot dwf=0.126
//sensor+printed_roller=0.241
//sensor+metal_roller=0.233
double dwf=0.233;

class ManipulabilityClient : public rclcpp::Node
{
public:
    using PointCloudMatrix = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;
    struct PointCloudAdaptor {
        const PointCloudMatrix &obj;
        PointCloudAdaptor(const PointCloudMatrix &obj_) : obj(obj_) {}
        inline size_t kdtree_get_point_count() const { return obj.rows(); }
        inline float kdtree_get_pt(const size_t idx, int dim) const { return obj(idx, dim); }
        template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
    };
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
        PointCloudAdaptor, 3>;
    // --- 主要な中間値や計算結果 ---
    std::vector<double> last_position_;
    Eigen::VectorXd joints_;
    Eigen::Matrix<double,4,4> FK_;
    Eigen::Matrix<double, 6, 7> J_;
    Eigen::Matrix<double, 7, 6> J_inv_;
    double manip_;
    Eigen::Matrix<double, 6, 7> Jq1_, Jq2_, Jq3_, Jq4_, Jq5_, Jq6_, Jq7_;
    Eigen::VectorXd trace_vec_;
    Eigen::VectorXd manipulability_gradient_;
    Eigen::Matrix<double, 6, 7> J_trans_;
    Eigen::VectorXd manipulability_trans_;
    bool execution_;
    Eigen::Vector3d destination_;
    Eigen::VectorXd movement_;

    // --- JointTrajectoryClient関連 ---
    rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::SharedPtr joint_trajectory_action_client_;
    rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr switch_controller_client_;
    rclcpp::Client<controller_manager_msgs::srv::LoadController>::SharedPtr load_controller_client_;
    rclcpp::Client<controller_manager_msgs::srv::ConfigureController>::SharedPtr configure_controller_client_;

    ManipulabilityClient()
    : Node("manipulability_client"), mapping_initialized_(false),
    got_pointcloud_(false),
    got_marker1_(false),
    got_marker2_(false),
    kd_tree_built_(false)
    {
        desired_order_ = {
            "lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4", "lbr_A5", "lbr_A6", "lbr_A7"
        };

        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/lbr/joint_states", 10,
            std::bind(&ManipulabilityClient::topic_callback, this, std::placeholders::_1));

        manip_pub_ = this->create_publisher<std_msgs::msg::Float64>("manipulability", 10);
        manip_trans_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("manipulability_trans", 10);
        fk_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("fk_matrix", 10);

        execution_ = false;
        destination_ = Eigen::Vector3d::Zero();
        movement_ = Eigen::VectorXd::Zero(6);
        execution_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/execution", 10,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                execution_ = msg->data;
            }
        );
        destination_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/destination", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() >= 3) {
                    for (int i = 0; i < 3; ++i)
                        destination_(i) = msg->data[i];
                }
            }
        );
        movement_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/movement", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() >= 6) {
                    for (int i = 0; i < 6; ++i)
                        movement_(i) = msg->data[i];
                }
            }
        );
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/capsule_cloud_transformed", 10,
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                if (!got_pointcloud_) {
                    parse_pointcloud(*msg);
                    got_pointcloud_ = true;
                    RCLCPP_INFO(this->get_logger(), "PointCloud received!");

                    // 停止合図パブリッシュ
                    std_msgs::msg::Bool sig;
                    sig.data = true;
                    pointcloud_acquired_pub_->publish(sig);
                }
            });

        marker1_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/marker1_transformed", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                if (!got_marker1_) {
                    marker1_pos_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
                    got_marker1_ = true;
                    RCLCPP_INFO(this->get_logger(), "marker1 received!");
                }
            });

        marker2_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/marker2_transformed", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
                if (!got_marker2_) {
                    marker2_pos_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
                    got_marker2_ = true;
                    RCLCPP_INFO(this->get_logger(), "marker2 received!");
                }
            });

        execution_pub_ = this->create_publisher<std_msgs::msg::Bool>("/execution", 1);
        pointcloud_acquired_pub_ = this->create_publisher<std_msgs::msg::Bool>("/pointcloud_acquired", 1);

        leptrino_sub_ = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            "/leptrino", 10,
            [this](const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
                leptrino_force_torque.header.stamp=msg->header.stamp;
                leptrino_force_torque.header.frame_id = "leptrino_sensor";
                leptrino_force_torque.wrench.force.x = msg->wrench.force.x;
                leptrino_force_torque.wrench.force.y = msg->wrench.force.y;
                leptrino_force_torque.wrench.force.z = msg->wrench.force.z;
                leptrino_force_torque.wrench.torque.x = msg->wrench.torque.x;
                leptrino_force_torque.wrench.torque.y = msg->wrench.torque.y;
                leptrino_force_torque.wrench.torque.z = msg->wrench.torque.z;
            });

        // --- JointTrajectoryClient 初期化 ---
        joint_trajectory_action_client_ =
            rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(
                this, "/lbr/joint_trajectory_controller/follow_joint_trajectory");

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

        check_pub_ = this->create_publisher<std_msgs::msg::Float64>("check", 10);
        setinitialposition_pub_ = this->create_publisher<std_msgs::msg::Bool>("/setinitialposition", 1);
    }
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr check_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr setinitialposition_pub_;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr execution_pub_;

    bool got_pointcloud() const { return got_pointcloud_; }
    const sensor_msgs::msg::PointCloud2 & get_pointcloud() const { return saved_cloud_; }
    bool got_marker1() const { return got_marker1_; }
    bool got_marker2() const { return got_marker2_; }
    Eigen::Vector3d get_marker1() const { return marker1_pos_; }
    Eigen::Vector3d get_marker2() const { return marker2_pos_; }

    geometry_msgs::msg::WrenchStamped get_force_torque() const {return leptrino_force_torque;}

    Eigen::Vector3d potential(const Eigen::Vector3d& current_pos, const Eigen::Vector3d& goal_pos, const double distance) {
        Eigen::Vector3d goal_vec;
        double goal_norm = 1.0 / sqrt(distance);

        Eigen::Vector3d calc_pos = current_pos;
        calc_pos[0] += 0.01;
        goal_vec = goal_pos - calc_pos;
        double goal_x_norm = 1.0 / sqrt(goal_vec.norm());

        calc_pos = current_pos;
        calc_pos[1] += 0.01;
        goal_vec = goal_pos - calc_pos;
        double goal_y_norm = 1.0 / sqrt(goal_vec.norm());

        calc_pos = current_pos;
        calc_pos[2] += 0.01;
        goal_vec = goal_pos - calc_pos;
        double goal_z_norm = 1.0 / sqrt(goal_vec.norm());

        Eigen::Vector3d direc;
        direc[0] = goal_x_norm - goal_norm;
        direc[1] = goal_y_norm - goal_norm;
        direc[2] = goal_z_norm - goal_norm;

        return direc;
    }

    Eigen::Vector3d manip_potential(const Eigen::VectorXd& joint, const Eigen::Matrix<double, 7, 6> Jinv, const double manip) {
        Eigen::Matrix<double, 6,1> movement;
        movement<<0.01,0,0,0,0,0;

        Eigen::VectorXd joint_x=joint+Jinv*movement;

        Eigen::Matrix<double, 6, 7> J_x = inversekinematics::calcJacobian(joint_x,dwf);
        double manip_x = manipulability::calcmanipulability(J_x);

        movement<<0,0.01,0,0,0,0;

        Eigen::VectorXd joint_y=joint+Jinv*movement;

        Eigen::Matrix<double, 6, 7> J_y = inversekinematics::calcJacobian(joint_y,dwf);
        double manip_y = manipulability::calcmanipulability(J_y);

        Eigen::Vector3d direc;
        direc[0] = manip_x - manip;
        direc[1] = manip_y - manip;
        direc[2] = manip - manip;

        return direc;
    }

    void parse_pointcloud(const sensor_msgs::msg::PointCloud2 & msg) {
        size_t n = msg.width * msg.height;
        saved_cloud_mat_.resize(n, 3);
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
        for (size_t i = 0; i < n; ++i, ++iter_x, ++iter_y, ++iter_z) {
            saved_cloud_mat_(i, 0) = *iter_x;
            saved_cloud_mat_(i, 1) = *iter_y;
            saved_cloud_mat_(i, 2) = *iter_z;
        }
        // KD-Tree構築
        adaptor_ = std::make_unique<PointCloudAdaptor>(saved_cloud_mat_);
        kd_tree_ = std::make_unique<KDTree>(3, *adaptor_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        kd_tree_->buildIndex();
        kd_tree_built_ = true;
    }

    // nanoflannによる高速最近傍探索
    Eigen::Vector3d get_nearest_point3(const Eigen::Vector3d& next_pos, double roi = 0.05) const {
    if (!got_pointcloud_ || !kd_tree_built_) return Eigen::Vector3d::Zero();
    float query_pt[3] = {static_cast<float>(next_pos.x()), static_cast<float>(next_pos.y()), static_cast<float>(next_pos.z())};
    const float search_radius = roi * roi; // nanoflannは距離の2乗
    // std::vector<std::pair<size_t, float>> ret_matches;
    std::vector<nanoflann::ResultItem<unsigned int, float>> ret_matches;
    nanoflann::SearchParameters params;
    kd_tree_->radiusSearch(query_pt, search_radius, ret_matches, params);

    if (ret_matches.empty()) {
        // ROI内に点がなければグローバル最近傍
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        kd_tree_->findNeighbors(resultSet, query_pt, nanoflann::SearchParameters(10));
        return saved_cloud_mat_.row(ret_index).cast<double>();
    }

    // ROI内でnext_posに最も近い点を探す（距離2乗が最小のもの）
    size_t best_idx = ret_matches[0].first;
    float best_dist = ret_matches[0].second;
    for (const auto& match : ret_matches) {
        if (match.second < best_dist) {
            best_idx = match.first;
            best_dist = match.second;
        }
    }
    return saved_cloud_mat_.row(best_idx).cast<double>();
    }

    void process()
    {
        if (last_position_.empty()) return;

        joints_ = Eigen::VectorXd::Map(last_position_.data(), last_position_.size());
        FK_ = forwardkinematics::calcfk(joints_,dwf);
        J_ = inversekinematics::calcJacobian(joints_,dwf);
        J_inv_ = inversekinematics::calcJacobianInverse(J_);

        manip_ = manipulability::calcmanipulability(J_);

        Jq1_ = manipulability::Jq1(joints_,dwf);
        Jq2_ = manipulability::Jq2(joints_,dwf);
        Jq3_ = manipulability::Jq3(joints_,dwf);
        Jq4_ = manipulability::Jq4(joints_,dwf);
        Jq5_ = manipulability::Jq5(joints_,dwf);
        Jq6_ = manipulability::Jq6(joints_,dwf);
        Jq7_ = manipulability::Jq7(joints_,dwf);

        trace_vec_ = manipulability::calctrace(Jq1_, Jq2_, Jq3_, Jq4_, Jq5_, Jq6_, Jq7_, J_inv_);
        manipulability_gradient_ = manipulability::gradient(manip_, trace_vec_);

        J_trans_ = inversekinematics::Jacobian_trans(J_);
        manipulability_trans_ = J_trans_ * manipulability_gradient_;

        // パブリッシュ
        std_msgs::msg::Float64 manip_msg;
        manip_msg.data = manip_;
        manip_pub_->publish(manip_msg);

        std_msgs::msg::Float64MultiArray manip_trans_msg;
        manip_trans_msg.data.resize(manipulability_trans_.size());
        for (int i = 0; i < manipulability_trans_.size(); ++i) {
            manip_trans_msg.data[i] = manipulability_trans_(i);
        }
        manip_trans_pub_->publish(manip_trans_msg);

        std_msgs::msg::Float64MultiArray fk_msg;
        fk_msg.data.resize(16); // 4x4=16
        for(int row=0; row<4; ++row) {
            for(int col=0; col<4; ++col) {
                fk_msg.data[row*4 + col] = FK_(row, col);
            }
        }
        fk_pub_->publish(fk_msg);
    }

    void process2(const Eigen::VectorXd &joints)
    {
        // if (last_position_.empty()) return;

        joints_ = joints;
        FK_ = forwardkinematics::calcfk(joints_,dwf);
        J_ = inversekinematics::calcJacobian(joints_,dwf);
        J_inv_ = inversekinematics::calcJacobianInverse(J_);

        manip_ = manipulability::calcmanipulability(J_);

        Jq1_ = manipulability::Jq1(joints_,dwf);
        Jq2_ = manipulability::Jq2(joints_,dwf);
        Jq3_ = manipulability::Jq3(joints_,dwf);
        Jq4_ = manipulability::Jq4(joints_,dwf);
        Jq5_ = manipulability::Jq5(joints_,dwf);
        Jq6_ = manipulability::Jq6(joints_,dwf);
        Jq7_ = manipulability::Jq7(joints_,dwf);

        trace_vec_ = manipulability::calctrace(Jq1_, Jq2_, Jq3_, Jq4_, Jq5_, Jq6_, Jq7_, J_inv_);
        manipulability_gradient_ = manipulability::gradient(manip_, trace_vec_);

        J_trans_ = inversekinematics::Jacobian_trans(J_);
        manipulability_trans_ = J_trans_ * manipulability_gradient_;

        // パブリッシュ
        std_msgs::msg::Float64 manip_msg;
        manip_msg.data = manip_;
        manip_pub_->publish(manip_msg);

        std_msgs::msg::Float64MultiArray manip_trans_msg;
        manip_trans_msg.data.resize(manipulability_trans_.size());
        for (int i = 0; i < manipulability_trans_.size(); ++i) {
            manip_trans_msg.data[i] = manipulability_trans_(i);
        }
        manip_trans_pub_->publish(manip_trans_msg);

        std_msgs::msg::Float64MultiArray fk_msg;
        fk_msg.data.resize(16); // 4x4=16
        for(int row=0; row<4; ++row) {
            for(int col=0; col<4; ++col) {
                fk_msg.data[row*4 + col] = FK_(row, col);
            }
        }
        fk_pub_->publish(fk_msg);
        // std::cout<<FK_<<std::endl;
    }

    void execute(const std::vector<double> &positions, const int32_t &sec_from_start = 10) {
        if (positions.size() != 7) { // KUKA::FRI::LBRState::NUMBER_OF_JOINTSを使う場合は置換
            RCLCPP_ERROR(this->get_logger(), "Invalid number of joint positions.");
            return;
        }

        control_msgs::action::FollowJointTrajectory::Goal joint_trajectory_goal;
        int32_t goal_sec_tolerance = 1;
        joint_trajectory_goal.goal_time_tolerance.sec = goal_sec_tolerance;

        trajectory_msgs::msg::JointTrajectoryPoint point;
        point.positions = positions;
        point.velocities.resize(7, 0.0);
        point.time_from_start.sec = sec_from_start;

        for (std::size_t i = 0; i < 7; ++i) {
            joint_trajectory_goal.trajectory.joint_names.push_back("lbr_A" + std::to_string(i + 1));
        }

        joint_trajectory_goal.trajectory.points.push_back(point);

        auto goal_future = joint_trajectory_action_client_->async_send_goal(joint_trajectory_goal);
        rclcpp::spin_until_future_complete(this->get_node_base_interface(), goal_future);
        auto goal_handle = goal_future.get();
        if (!goal_handle) {
            RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server.");
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Goal was accepted by server.");

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
        request->strictness = 2;

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
        request->strictness = 2;

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

    void topic_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        if (!mapping_initialized_) {
            index_map_.resize(desired_order_.size());
            for (size_t i = 0; i < desired_order_.size(); ++i) {
                auto it = std::find(msg->name.begin(), msg->name.end(), desired_order_[i]);
                if (it != msg->name.end()) {
                    index_map_[i] = std::distance(msg->name.begin(), it);
                } else {
                    index_map_[i] = -1;
                }
            }
            mapping_initialized_ = true;
        }
        last_position_.resize(desired_order_.size());
        for (size_t i = 0; i < index_map_.size(); ++i) {
            if (index_map_[i] >= 0 && (size_t)index_map_[i] < msg->position.size()) {
                last_position_[i] = msg->position[index_map_[i]];
            } else {
                last_position_[i] = 0.0;
            }
        }
    }

    // --- 各種パブリッシャ・サブスクライバ ---
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manip_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr manip_trans_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr fk_pub_;
    std::vector<std::string> desired_order_;
    std::vector<int> index_map_;
    bool mapping_initialized_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr execution_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr destination_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr movement_sub_;

    // Eigen::MatrixXf saved_cloud_mat_; // Nx3
    Eigen::Vector3d marker1_pos_;
    Eigen::Vector3d marker2_pos_;
    bool got_pointcloud_;
    bool got_marker1_, got_marker2_;
    sensor_msgs::msg::PointCloud2 saved_cloud_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pointcloud_acquired_pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr marker1_sub_, marker2_sub_;

    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr check_sub_;
    double check_;
    rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr leptrino_sub_;
    geometry_msgs::msg::WrenchStamped leptrino_force_torque;

    PointCloudMatrix saved_cloud_mat_; // Nx3
    mutable std::unique_ptr<PointCloudAdaptor> adaptor_;
    mutable std::unique_ptr<KDTree> kd_tree_;
    bool kd_tree_built_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManipulabilityClient>();
    rclcpp::Rate rate(100);  // 100Hz

    bool prev_execution = false;
    int count=0;

    RCLCPP_INFO(node->get_logger(), "goes to initial position");
    std::vector<double> positions = {deg2rad(18.62),deg2rad(93.26),deg2rad(110.67),deg2rad(-75.41),deg2rad(-117.52),deg2rad(-91.32),deg2rad(-3.29)};
    // node->execute({deg2rad(-20.0),deg2rad(80.0),deg2rad(100.0),deg2rad(-80.0),deg2rad(-80.0),deg2rad(-80.0),deg2rad(20.0),});
    // node->execute({deg2rad(18.62),deg2rad(93.26),deg2rad(110.67),deg2rad(-75.41),deg2rad(-117.52),deg2rad(-91.32),deg2rad(-3.29),},6);
    node->execute(positions,6);
    std_msgs::msg::Bool setinitialposition_msg;
    setinitialposition_msg.data=true;
    node->setinitialposition_pub_->publish(setinitialposition_msg);

    std_msgs::msg::Bool execution_msg;
    execution_msg.data=false;
    Eigen::Vector3d start_pos;
    start_pos<<0.02,0.393,0.8;
    Eigen::Vector3d goal_pos;
    goal_pos<<0.42,0.393,0.8;

    geometry_msgs::msg::WrenchStamped force_torque;

    while (rclcpp::ok() && (!node->got_pointcloud() || !node->got_marker1() || !node->got_marker2())) {
        rclcpp::spin_some(node);
        node->process();
        node->setinitialposition_pub_->publish(setinitialposition_msg);
        node->execution_pub_->publish(execution_msg);

        if(count==0) std::cout<<"waiting for pointcloud "<<std::endl;
        count++;

        rate.sleep();
    }

    start_pos=node->get_marker1();
    start_pos=node->get_nearest_point3(start_pos,0.05);
    goal_pos=node->get_marker2();
    goal_pos=node->get_nearest_point3(goal_pos,0.05);

    Eigen::Vector3d dest=start_pos;
    Eigen::Matrix<double,4,4> FK=node->FK_;
    Eigen::Vector3d current;
    current<<FK(0,3),FK(1,3),FK(2,3);

    Eigen::Vector3d start_vec=start_pos-current;
    double start_dist=start_vec.norm();
    start_vec=goal_pos-current;
    if(start_dist>start_vec.norm()){
        dest=start_pos;
        start_pos=goal_pos;
        goal_pos=dest;
    }

    count=0;

    dest=start_pos;
    Eigen::VectorXd joints=node->joints_;
    Eigen::Matrix<double, 6, 7> J=node->J_;
    Eigen::Matrix<double, 7, 6> J_inv=node->J_inv_;

    Eigen::Matrix<double, 7, 1> q_dt;

    Eigen::Vector3d z_angle;
    z_angle<<FK(0,2),FK(1,2),FK(2,2);
    Eigen::Vector3d z_refer;
    z_refer<<0,0,1;
    Eigen::Vector3d rotation;
    rotation<<0,0,0;
    double stepsize=0.001;
    double rotang=0.0;
    Eigen::Vector3d rotax;
    rotax<<0,0,0;

    // double dwf=0.126;

    //開始位置に到達する角度を計算
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->process();

        FK = forwardkinematics::calcfk(joints,dwf);
        J = inversekinematics::calcJacobian(joints,dwf);
        J_inv = inversekinematics::calcJacobianInverse(J);
        current<<FK(0,3),FK(1,3),FK(2,3);

        Eigen::Vector3d direction=dest-current;
        double distance=direction.norm();
        if(distance<0.005){
            break;
        }
        direction=stepsize*direction/distance;
        if(count==0) {
            std::cout<<"calculating start position ("<<dest(0)<<"  "<<dest(1)<<"  "<<dest(2)<<")"<<std::endl;
            rotax = z_angle.cross(z_refer);
            rotax = rotax / rotax.norm();
            rotang = acos(z_refer.dot(z_angle));
            rotang=rotang/(distance/stepsize);
        }
        rotation = rotax * rotang;
        Eigen::Matrix<double, 6, 1> move;
        move<<direction,rotation;

        q_dt=J_inv*move;
        joints[0]+=q_dt(0,0);
        joints[1]+=q_dt(1,0);
        joints[2]+=q_dt(2,0);
        joints[3]+=q_dt(3,0);
        joints[4]+=q_dt(4,0);
        joints[5]+=q_dt(5,0);
        joints[6]+=q_dt(6,0);

        count++;

        rate.sleep();
    }
    count=0;

    ///位置制御を実行
    RCLCPP_INFO(node->get_logger(), "goes to start position");
    node->execute({
        joints[0],
        joints[1],
        joints[2],
        joints[3],
        joints[4],
        joints[5],
        joints[6],
    },5);

    double coef_manip=0;//30
    double coef_pos=1.0;

    double target_force=3.0;
    double error=0, error_before=0;
    double ie=0,id=0;
    double kp=0.00005,ki=0.0,kd=0.0;
    double surface_modify=0.0;
    double surface_modify_save=0.0;

    Eigen::Vector3d goal_pos_save=goal_pos;
    Eigen::Vector3d current_pos;

    int CONTROL_HZ=100;

    int split_count=5;
    int move_count=0;
    Eigen::Vector3d nearest_move=Eigen::Vector3d::Zero();
    Eigen::Vector3d move_step=Eigen::Vector3d::Zero();

    //追従制御開始
    rclcpp::Rate rate2(CONTROL_HZ);  // 100Hz

    std::string input;
    std::cout<<"Execute ? y/n"<<std::endl;
    std::cin>> input;
    node->process2(joints);

    if(input=="y"){
        std::ofstream fo("/home/isrlab/colcon_ws/src/motion_planning/src/data.csv");
        fo<<"joint1,2,3,4,5,6,7,manip,x,y,z,force"<<std::endl;

        while (rclcpp::ok()) {
            rclcpp::spin_some(node);
            joints=node->joints_;
            J_inv=node->J_inv_;
            FK=node->FK_;

            //////////////////////////////////////////////
            force_torque=node->get_force_torque();

            Eigen::Vector3d z_current;
            z_current<<FK(0,2),FK(1,2),FK(2,2);

            Eigen::VectorXd manip_trans = node->manipulability_trans_;

            // Eigen::Vector3d manip_direc = manip_trans.head<3>();
            // current_pos<<FK(0,3),FK(1,3),FK(2,3);
            // goal_pos=goal_pos_save+z_current*surface_modify;
            // Eigen::Vector3d goal_vec=goal_pos-current_pos;
            // double distance=goal_vec.norm();

            error = target_force-abs(force_torque.wrench.force.z);
            std::cout<<"tuch error: "<<error<<std::endl;
            error=0;
            if(error<0.5){
                break;
            }
            // error=0;
            // if(count<20) error = 1.5;
            // if(count>20) error = 1.0;
            // if(count>30) error = 0.5;
            // if(count>40) error = 0.0;
            ie += error/CONTROL_HZ;
            id = (error-error_before)/CONTROL_HZ;
            error_before = error;
            surface_modify_save=surface_modify;
            surface_modify += (error*kp+ie*ki+id*kd);
            std::cout<<"surface_modify: "<<surface_modify<<std::endl;

            if(abs(surface_modify)>0.05){
                std::cout<<"stop due to pointcloud or force error"<<std::endl;
                break;
            }

            Eigen::Matrix<double, 6,1> movement;
            // movement<<nearest_move,0,0,0;
            movement<<z_current*(surface_modify-surface_modify_save),0,0,0;

            // std::cout<<"current:    "<<current_pos.transpose()<<std::endl;
            // std::cout<<"error: "<<error<<std::endl;
            // std::cout<<"surface_modify: "<<surface_modify<<std::endl;
            // std::cout<<"count: "<<count<<std::endl;
            // std::cout<<"movement "<<movement.transpose()<<std::endl;
            // std::cout<<"distance "<<distance<<std::endl;

            ///////////////////////////////////////////////////////////////////////

            if(abs(force_torque.wrench.force.z)>5){
                for (int i = 0; i < 6; ++i) {
                    movement[i] = 0;
                }
                std::cout<<"much pressure!"<<std::endl;
            }
            
            q_dt=J_inv*movement;
            joints[0]+=q_dt(0,0);
            joints[1]+=q_dt(1,0);
            joints[2]+=q_dt(2,0);
            joints[3]+=q_dt(3,0);
            joints[4]+=q_dt(4,0);
            joints[5]+=q_dt(5,0);
            joints[6]+=q_dt(6,0);


            node->execute({joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],joints[6]},1.5);
            // node->process();
            node->process2(joints);
            count++;

            // rate2.sleep();
        }
        error=0, error_before=0;
        ie=0,id=0;
        count=0;
        std::cout<<"track start"<<std::endl;

        // while (rclcpp::ok()) {
        //     std::chrono::system_clock::time_point start, end;
        //     start = std::chrono::system_clock::now();

        //     rclcpp::spin_some(node);
        //     joints=node->joints_;
        //     J_inv=node->J_inv_;
        //     FK=node->FK_;

        //     //////////////////////////////////////////////
        //     force_torque=node->get_force_torque();

        //     Eigen::Vector3d z_current;
        //     z_current<<FK(0,2),FK(1,2),FK(2,2);

        //     Eigen::VectorXd manip_trans = node->manipulability_trans_;

        //     Eigen::Vector3d manip_direc = manip_trans.head<3>();
        //     current_pos<<FK(0,3),FK(1,3),FK(2,3);
        //     goal_pos=goal_pos_save+z_current*surface_modify;
        //     Eigen::Vector3d goal_vec=goal_pos-current_pos;
        //     double distance=goal_vec.norm();

        //     error = target_force-abs(force_torque.wrench.force.z);
        //     // error=0;
        //     // if(count<20) error = 1.5;
        //     // if(count>20) error = 1.0;
        //     // if(count>30) error = 0.5;
        //     // if(count>40) error = 0.0;
        //     ie += error/CONTROL_HZ;
        //     id = (error-error_before)/CONTROL_HZ;
        //     error_before = error;
        //     surface_modify += error*kp+ie*ki+id*kd;

        //     if(abs(surface_modify)>0.05){
        //         std::cout<<"stop due to pointcloud or force error"<<std::endl;
        //         break;
        //     }

        //     Eigen::Vector3d goal_pot=node->potential(current_pos, goal_pos,distance);
        //     Eigen::Vector3d direction=coef_manip*manip_direc+coef_pos*goal_pot;

        //     double stepsize=0.003;
        //     Eigen::Vector3d move_trans=stepsize*direction/direction.norm();
        //     Eigen::Vector3d next_pos=current_pos-z_current*surface_modify+move_trans;
        //     Eigen::Vector3d nearest_point = node->get_nearest_point3(next_pos,0.05);
        //     Eigen::Vector3d nearest_move=nearest_point-current_pos;

        //     Eigen::Matrix<double, 6,1> movement;
        //     // movement<<nearest_move,0,0,0;
        //     movement<<nearest_move+z_current*surface_modify,0,0,0;

        //     // std::cout<<"current:    "<<current_pos.transpose()<<std::endl;
        //     std::cout<<"surface_modify: "<<surface_modify<<std::endl;
        //     std::cout<<"count: "<<count<<std::endl;
        //     // std::cout<<"movement "<<movement.transpose()<<std::endl;
        //     std::cout<<"distance "<<distance<<std::endl;

        //     if (distance<0.01){
        //         for (int i = 0; i < 6; ++i) {
        //             movement[i] = 0;
        //         }
        //         std::cout<<"end     x: "<<current_pos(0)<<"  y: "<<current_pos(1)<<"  z: "<<current_pos(2)<<std::endl;
        //         std::cout<<"goal: "<<goal_pos.transpose()<<std::endl;
        //         break;
        //     }
        //     ///////////////////////////////////////////////////////////////////////

        //     if(abs(force_torque.wrench.force.z)>5){
        //         for (int i = 0; i < 6; ++i) {
        //             movement[i] = 0;
        //         }
        //         std::cout<<"much pressure!"<<std::endl;
        //     }
            
        //     q_dt=J_inv*movement;
        //     joints[0]+=q_dt(0,0);
        //     joints[1]+=q_dt(1,0);
        //     joints[2]+=q_dt(2,0);
        //     joints[3]+=q_dt(3,0);
        //     joints[4]+=q_dt(4,0);
        //     joints[5]+=q_dt(5,0);
        //     joints[6]+=q_dt(6,0);
        //     // if(count<1) node->execute({joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],joints[6]},1);
        //     // if(count>10) node->execute({joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],joints[6]},0.2);
        //     // node->execute({joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],joints[6]},0.8);
        //     // node->process();
        //     node->process2(joints);
        //     count++;

        //     end = std::chrono::system_clock::now();
        //     double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
        //     printf("time %lf[ms]\n", time);

        //     rate2.sleep();
        // }
        count=0;

        while (rclcpp::ok()) {
            rclcpp::spin_some(node);
            joints=node->joints_;
            J_inv=node->J_inv_;
            FK=node->FK_;
            double manip=node->manip_;

            //////////////////////////////////////////////
            force_torque=node->get_force_torque();

            fo<<joints[0]<<","<<joints[1]<<","<<joints[2]<<","<<joints[3]<<","<<joints[4]<<","<<joints[5]<<","<<joints[6]<<","<<manip<<","<<FK(0,3)<<","<<FK(1,3)<<","<<FK(2,3)<<","<<force_torque.wrench.force.z<<","<<std::endl;

            Eigen::Vector3d z_current;
            z_current<<FK(0,2),FK(1,2),FK(2,2);

            // Eigen::VectorXd manip_trans = node->manipulability_trans_;
            // Eigen::Vector3d manip_direc = manip_trans.head<3>();
            Eigen::Vector3d manip_direc = node->manip_potential(joints, J_inv,manip);
            current_pos<<FK(0,3),FK(1,3),FK(2,3);
            goal_pos=goal_pos_save+z_current*surface_modify;
            Eigen::Vector3d goal_vec=goal_pos-current_pos;
            double distance=goal_vec.norm();

            error = target_force-abs(force_torque.wrench.force.z);
            error=0;
            // if(count<20) error = 1.5;
            // if(count>20) error = 1.0;
            // if(count>30) error = 0.5;
            // if(count>40) error = 0.0;
            ie += error/CONTROL_HZ;
            id = (error-error_before)/CONTROL_HZ;
            error_before = error;
            surface_modify_save=surface_modify;
            surface_modify += error*kp+ie*ki+id*kd;

            if(abs(surface_modify)>0.05){
                std::cout<<"stop due to pointcloud or force error"<<std::endl;
                break;
            }

            Eigen::Vector3d goal_pot=node->potential(current_pos, goal_pos,distance);
            Eigen::Vector3d direction=coef_manip*manip_direc+coef_pos*goal_pot;

            double stepsize=0.005;
            Eigen::Vector3d move_trans=stepsize*direction/direction.norm();
            Eigen::Vector3d next_pos=current_pos-z_current*surface_modify+move_trans;
            Eigen::Vector3d nearest_point = node->get_nearest_point3(next_pos,0.05);

            nearest_move=nearest_point-current_pos;
            nearest_move(2)=0;
            move_step=nearest_move/split_count;

            // if(move_count==0){
            //     nearest_move=nearest_point-current_pos;
            //     nearest_move(2)=0;
            //     move_step=nearest_move/split_count;
            // }
            // move_count++;
            // if(move_count>=split_count){
            //     move_count=0;
            // }

            // if(abs(force_torque.wrench.force.z)>10){
            //     for (int i = 0; i < 3; ++i) {
            //         nearest_move[i] = 0;
            //     }
            //     std::cout<<"much pressure!"<<std::endl;
            // }

            Eigen::Matrix<double, 6,1> movement;
            // movement<<nearest_move,0,0,0;
            // movement<<nearest_move+z_current*(surface_modify-surface_modify_save),0,0,0;
            movement<<move_step+z_current*(surface_modify-surface_modify_save),0,0,0;

            // std::cout<<"current:    "<<current_pos.transpose()<<std::endl;
            std::cout<<"error: "<<error<<std::endl;
            std::cout<<"surface_modify: "<<surface_modify<<std::endl;
            std::cout<<"count: "<<count<<std::endl;
            std::cout<<"distance "<<distance<<std::endl;

            if (distance<0.01){
                for (int i = 0; i < 6; ++i) {
                    movement[i] = 0;
                }
                std::cout<<"end     x: "<<current_pos(0)<<"  y: "<<current_pos(1)<<"  z: "<<current_pos(2)<<std::endl;
                std::cout<<"goal: "<<goal_pos.transpose()<<std::endl;
                break;
            }
            ///////////////////////////////////////////////////////////////////////

            if(abs(force_torque.wrench.force.z)>30){
                for (int i = 0; i < 3; ++i) {
                    movement[i] = 0;
                }
                std::cout<<"much pressure!"<<std::endl;
            }
            
            q_dt=J_inv*movement;
            joints[0]+=q_dt(0,0);
            joints[1]+=q_dt(1,0);
            joints[2]+=q_dt(2,0);
            joints[3]+=q_dt(3,0);
            joints[4]+=q_dt(4,0);
            joints[5]+=q_dt(5,0);
            joints[6]+=q_dt(6,0);


            node->execute({joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],joints[6]},0.7);
            node->process();
            // node->process2(joints);
            count++;

            if(count>700){
                break;
            }

            // rate2.sleep();
        }
    }

    node->execute(positions,6);

    rclcpp::shutdown();
    return 0;
}
