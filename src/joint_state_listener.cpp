#include <chrono>
#include <memory>
#include <vector>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include "eigen3/Eigen/Dense"
#include "motion_planning/inverse.h"
#include "motion_planning/manipulability_gradient.h"

using namespace std::chrono_literals;

class JointStateListener : public rclcpp::Node
{
public:
    JointStateListener()
    : Node("joint_state_listener")
    {
        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/lbr/joint_states", 10,
            std::bind(&JointStateListener::topic_callback, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(10ms, std::bind(&JointStateListener::timer_callback, this));
    }

private:
    void topic_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        last_position_.resize(msg->position.size());
        for (size_t i = 0; i < msg->position.size(); ++i)
            last_position_[i] = msg->position[i];
    }

    void timer_callback()
    {
        if (last_position_.empty()) return;

        // Eigenベクトルに変換
        Eigen::VectorXd joints = Eigen::VectorXd::Map(last_position_.data(), last_position_.size());
        // ヤコビ行列を計算
        Eigen::Matrix<double, 6, 7> J = inversekinematics::calcJacobian(joints);
        // 逆行列も計算
        Eigen::Matrix<double, 7, 6> J_inv = inversekinematics::calcJacobianInverse(J);

        double manip=manipulability::calcmanipulability(J);

        Eigen::Matrix<double, 6, 7> Jq1=manipulability::Jq1(joints);
        Eigen::Matrix<double, 6, 7> Jq2=manipulability::Jq2(joints);
        Eigen::Matrix<double, 6, 7> Jq3=manipulability::Jq3(joints);
        Eigen::Matrix<double, 6, 7> Jq4=manipulability::Jq4(joints);
        Eigen::Matrix<double, 6, 7> Jq5=manipulability::Jq5(joints);
        Eigen::Matrix<double, 6, 7> Jq6=manipulability::Jq6(joints);
        Eigen::Matrix<double, 6, 7> Jq7=manipulability::Jq7(joints);

        Eigen::VectorXd trace_vec=manipulability::calctrace(Jq1,Jq2,Jq3,Jq4,Jq5,Jq6,Jq7,J_inv);
        Eigen::VectorXd manipulability_gradient=manipulability::gradient(manip,trace_vec);

        Eigen::Matrix<double, 6, 7> J_trans=inversekinematics::Jacobian_trans(J);

        //Eigen::Matrix<double, 7, 6> J_trans_inv = inversekinematics::calcJacobianInverse(J_trans);

        std::cout << "Jacobian:\n" << J << std::endl;
        std::cout << "Jacobian Inverse:\n" << J_inv << std::endl;
        std::cout<<"manipulability:\n"<<manip<<std::endl;
        std::cout << "trace:\n" << trace_vec << std::endl;
        std::cout << "gradient:\n" << manipulability_gradient << std::endl;
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<double> last_position_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JointStateListener>());
    rclcpp::shutdown();
    return 0;
}
