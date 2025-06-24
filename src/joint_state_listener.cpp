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

class JointStateListener : public rclcpp::Node
{
public:
    JointStateListener()
    : Node("joint_state_listener"), mapping_initialized_(false)
    {
        // 希望する順番を記述（例: A1～A7）
        desired_order_ = {
            "lbr_A1",
            "lbr_A2",
            "lbr_A3",
            "lbr_A4",
            "lbr_A5",
            "lbr_A6",
            "lbr_A7"
        };

        subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/lbr/joint_states", 10,
            std::bind(&JointStateListener::topic_callback, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(10ms, std::bind(&JointStateListener::timer_callback, this));

        manip_pub_ = this->create_publisher<std_msgs::msg::Float64>("manipulability", 10);
        manip_trans_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("manipulability_trans", 10);
        fk_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("fk_matrix", 10);

    }

private:
    void topic_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // 初回のみマッピング作成
        if (!mapping_initialized_) {
            index_map_.resize(desired_order_.size());
            for (size_t i = 0; i < desired_order_.size(); ++i) {
                auto it = std::find(msg->name.begin(), msg->name.end(), desired_order_[i]);
                if (it != msg->name.end()) {
                    index_map_[i] = std::distance(msg->name.begin(), it);
                } else {
                    index_map_[i] = -1;  // 該当ジョイントがなければ-1
                }
            }
            mapping_initialized_ = true;
        }

        // 並び替えたポジションを作成
        last_position_.resize(desired_order_.size());
        for (size_t i = 0; i < index_map_.size(); ++i) {
            if (index_map_[i] >= 0 && (size_t)index_map_[i] < msg->position.size()) {
                last_position_[i] = msg->position[index_map_[i]];
            } else {
                last_position_[i] = 0.0; // 該当がなければ0
            }
        }
    }

    void timer_callback()
    {
        if (last_position_.empty()) return;

        // 開始時間を記録
        //auto t_start = std::chrono::high_resolution_clock::now();

        // Eigenベクトルに変換
        Eigen::VectorXd joints = Eigen::VectorXd::Map(last_position_.data(), last_position_.size());
        //順運動学
        Eigen::Matrix<double,4,4> FK=forwardkinematics::calcfk(joints);
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

        Eigen::VectorXd manipulability_trans=J_trans*manipulability_gradient;

        //Eigen::Matrix<double, 7, 6> J_trans_inv = inversekinematics::calcJacobianInverse(J_trans);

        // 終了時間を記録
        //auto t_end = std::chrono::high_resolution_clock::now();

        // 経過時間をミリ秒で出力
        //auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

        //std::cout << "計算時間: " << elapsed << " マイクロ秒" << std::endl;

        std_msgs::msg::Float64 manip_msg;
        manip_msg.data = manip;
        manip_pub_->publish(manip_msg);

        // manipulability_transをパブリッシュ
        std_msgs::msg::Float64MultiArray manip_trans_msg;
        manip_trans_msg.data.resize(manipulability_trans.size());
        for (int i = 0; i < manipulability_trans.size(); ++i) {
            manip_trans_msg.data[i] = manipulability_trans(i);
        }
        manip_trans_pub_->publish(manip_trans_msg);

        std_msgs::msg::Float64MultiArray fk_msg;
        fk_msg.data.resize(16); // 4x4=16
        for(int row=0; row<4; ++row) {
            for(int col=0; col<4; ++col) {
                fk_msg.data[row*4 + col] = FK(row, col);
            }
        }
        fk_pub_->publish(fk_msg);

        //std::cout << "Jacobian:\n" << J << std::endl;
        //std::cout << "Jacobian Inverse:\n" << J_inv << std::endl;
        //std::cout<<"manipulability:\n"<<manip<<std::endl;
        //std::cout << "trace:\n" << trace_vec << std::endl;
        //std::cout << "gradient:\n" << manipulability_gradient << std::endl;
        //std::cout << "FK:\n" << FK << std::endl;
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<double> last_position_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr manip_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr manip_trans_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr fk_pub_;


    // 並び替え用メンバ
    std::vector<std::string> desired_order_;
    std::vector<int> index_map_;
    bool mapping_initialized_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<JointStateListener>());
    rclcpp::shutdown();
    return 0;
}
