#include <chrono>
#include <memory>
#include <vector>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include "eigen3/Eigen/Dense"

#include "motion_planning/inverse.h"
#include "motion_planning/manipulability_gradient.h"
#include "motion_planning/forward.h"

#include <algorithm>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

using namespace std::chrono_literals;

class ManipulabilityClient : public rclcpp::Node
{
public:
    // 主要な中間値や計算結果を全てpublicメンバ変数として宣言
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
        execution_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/execution", 10,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                execution_ = msg->data;
            });
        destination_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "/destination", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() >= 3) {
                    for (int i = 0; i < 3; ++i)
                        destination_(i) = msg->data[i];
                }
            });
    }

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

    // サブスクライバコールバック（このまま）
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

    // パブリッシャなど
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manip_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr manip_trans_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr fk_pub_;

    std::vector<std::string> desired_order_;
    std::vector<int> index_map_;
    bool mapping_initialized_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr execution_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr destination_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManipulabilityClient>();
    rclcpp::Rate rate(100);  // 100Hz

    bool prev_execution = false;

    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->process();

        if (node->execution_ && !prev_execution) {
            prev_execution=true;
            break;
        }
        prev_execution = node->execution_;

        rate.sleep();
    }
    Eigen::Vector3d dest=node->destination_;
    Eigen::VectorXd joints=node->joints_
    Eigen::Matrix<double,4,4> FK=node->FK_;
    Eigen::Vector3d current;
    current<<FK(0,3),FK(1,3),FK(2,3);
    Eigen::Matrix<double, 6, 7> J=node->J_;
    Eigen::Matrix<double, 7, 6> J_inv=node->J_inv_;

    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->process();

        FK = forwardkinematics::calcfk(joints);
        J = inversekinematics::calcJacobian(joints);
        J_inv = inversekinematics::calcJacobianInverse(J);
        current<<FK(0,3),FK(1,3),FK(2,3);

        Eigen::Vector3d direction=dest-current;
        double distance=norm(direction);
        if(distance<0.01){
            break;
        }
        direction=0.001*direction/distance;
        Eigen::Matrix<double, 6, 1> move;
        move<<direction,0,0,0;

        Eigen::Matrix<double, 7, 1> q_dt=J_inv*move;
        joints[0]+=q_dt(0,0);
        joints[1]+=q_dt(1,0);
        joints[2]+=q_dt(2,0);
        joints[3]+=q_dt(3,0);
        joints[4]+=q_dt(4,0);
        joints[5]+=q_dt(5,0);
        joints[6]+=q_dt(6,0);


        rate.sleep();
    }

    ///ここで位置制御

    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->process();

        if (!(node->execution_) && prev_execution) {
            prev_execution=false;
            break;
        }
        prev_execution = node->execution_;

        rate.sleep();
    }

    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->process();

        // main内のどこからでも変数を参照・出力できる！
        std::cout << "manip: " << node->manip_ << std::endl;
        std::cout << "J_inv:\n" << node->J_inv_ << std::endl;
        std::cout << "trace_vec: " << node->trace_vec_.transpose() << std::endl;
        // 必要に応じて他のメンバ変数も出力できます

        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
