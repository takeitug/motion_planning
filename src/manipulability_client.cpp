#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <algorithm>

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

#define PI 3.14159265358979323846
double rad2deg(double radians) {
    return radians * (180.0 / PI);
}
double deg2rad(double degrees) {
    return degrees * (PI / 180.0);
}

class ManipulabilityClient : public rclcpp::Node
{
public:
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
    : Node("manipulability_client"), mapping_initialized_(false)
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

    void process()
    {
        if (last_position_.empty()) return;

        joints_ = Eigen::VectorXd::Map(last_position_.data(), last_position_.size());
        FK_ = forwardkinematics::calcfk(joints_);
        J_ = inversekinematics::calcJacobian(joints_);
        J_inv_ = inversekinematics::calcJacobianInverse(J_);

        manip_ = manipulability::calcmanipulability(J_);

        Jq1_ = manipulability::Jq1(joints_);
        Jq2_ = manipulability::Jq2(joints_);
        Jq3_ = manipulability::Jq3(joints_);
        Jq4_ = manipulability::Jq4(joints_);
        Jq5_ = manipulability::Jq5(joints_);
        Jq6_ = manipulability::Jq6(joints_);
        Jq7_ = manipulability::Jq7(joints_);

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
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManipulabilityClient>();
    rclcpp::Rate rate(100);  // 100Hz

    bool prev_execution = false;
    int count=0;

    RCLCPP_INFO(node->get_logger(), "goes to initial position");
    // node->execute({deg2rad(-20.0),deg2rad(80.0),deg2rad(100.0),deg2rad(-80.0),deg2rad(-80.0),deg2rad(-80.0),deg2rad(20.0),});
    node->execute({deg2rad(18.62),deg2rad(93.26),deg2rad(110.67),deg2rad(-75.41),deg2rad(-117.52),deg2rad(-91.32),deg2rad(-3.29),});
    std_msgs::msg::Bool setinitialposition_msg;
    setinitialposition_msg.data=true;
    node->setinitialposition_pub_->publish(setinitialposition_msg);

    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->process();
        node->setinitialposition_pub_->publish(setinitialposition_msg);

        if (node->execution_ && !prev_execution) {
            //マーカーと点群が取得できたか判別
            prev_execution=true;
            break;
        }
        prev_execution = node->execution_;
        if(count==0) std::cout<<"waiting for pointcloud "<<std::endl;
        count++;

        rate.sleep();
    }
    count=0;

    Eigen::Vector3d dest=node->destination_;
    Eigen::VectorXd joints=node->joints_;
    Eigen::Matrix<double,4,4> FK=node->FK_;
    Eigen::Vector3d current;
    current<<FK(0,3),FK(1,3),FK(2,3);
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

    //開始位置に到達する角度を計算
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->process();
        dest=node->destination_;

        FK = forwardkinematics::calcfk(joints);
        J = inversekinematics::calcJacobian(joints);
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
    });

    //追従制御の開始合図を待つ
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->process();

        if (!(node->execution_) && prev_execution) {
            prev_execution=false;
            break;
        }
        prev_execution = node->execution_;
        if(count==0) std::cout<<"waiting for que "<<std::endl;
        count++;

        rate.sleep();
    }
    count=0;

    while (rclcpp::ok() && node->movement_ == Eigen::VectorXd::Zero(6)) {
        rclcpp::spin_some(node);
        node->process();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if(count==0) std::cout<<"waiting for movement "<<std::endl;
        count++;
    }
    count=0;
    Eigen::VectorXd movement=node->movement_;
    Eigen::VectorXd movement_save=node->movement_;

    // joints=node->joints_;
    // node->execute({
    //     joints[0],
    //     joints[1],
    //     joints[2],
    //     joints[3],
    //     joints[4],
    //     joints[5],
    //     joints[6],
    // });
    // std::this_thread::sleep_for(std::chrono::seconds(15));

    //追従制御開始
    rclcpp::Rate rate2(2);  // 100Hz
    while (rclcpp::ok()) {
        node->process();
        rclcpp::spin_some(node);
        joints=node->joints_;
        J_inv=node->J_inv_;
        movement=node->movement_;
        std::cout<<"movement "<<movement<<std::endl;
        // if(movement==movement_save){
        //     continue;
        // }

        Eigen::Vector3d movement_distance = movement.head<3>();
        
        q_dt=J_inv*movement;
        joints[0]+=q_dt(0,0);
        joints[1]+=q_dt(1,0);
        joints[2]+=q_dt(2,0);
        joints[3]+=q_dt(3,0);
        joints[4]+=q_dt(4,0);
        joints[5]+=q_dt(5,0);
        joints[6]+=q_dt(6,0);
        // if(count<1) node->execute({joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],joints[6]},1);
        // if(count>10) node->execute({joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],joints[6]},0.2);
        node->execute({joints[0],joints[1],joints[2],joints[3],joints[4],joints[5],joints[6]},0.2);
        std_msgs::msg::Float64 check_msg;
        check_msg.data=count;
        node->check_pub_->publish(check_msg);
        count++;

        // std::cout << "manip: " << node->manip_ << std::endl;
        // std::cout << "J_inv:\n" << node->J_inv_ << std::endl;
        // std::cout << "trace_vec: " << node->trace_vec_.transpose() << std::endl;
        if(node->execution_) break;

        rate2.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
