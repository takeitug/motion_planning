#include <memory>
#include <vector>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <array>

#include "eigen3/Eigen/Dense"

class ManipulabilityPlanner : public rclcpp::Node
{
public:
    ManipulabilityPlanner()
    : Node("manipulability_planner"), manip_(0.0), manip_trans_(6, 0.0),fk_mat_(16, 0.0), fk_col4_(4, 0.0)
    {
        manip_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "manipulability", 10,
            [this](const std_msgs::msg::Float64::SharedPtr msg) {
                manip_ = msg->data;
            });

        manip_trans_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "manipulability_trans", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                manip_trans_ = msg->data;
            });
        fk_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "fk_matrix", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                fk_mat_ = msg->data;
                // 4列目のみ抽出
                if (fk_mat_.size() == 16) {
                    for (int i = 0; i < 4; ++i) {
                        // 4列目: 各行の3列目 (row*4 + 3)
                        fk_col4_[i] = fk_mat_[i*4 + 3];
                    }
                }
            });
    }

    double get_manip() const { return manip_; }
    std::vector<double> get_manip_trans() const { return manip_trans_; }
    std::vector<double> get_fk_col4() const { return fk_col4_; }

    std::vector<double> potential(std::vector<double> manip, std::vector<double> current_pos, std::vector<double> goal_pos){
        std::vector<double> goal_vec=goal_pos-current_pos;
        double goal_norm=1/sqrt(goal_vec[0]*goal_vec[0]+goal_vec[1]*goal_vec[1]+goal_vec[2]*goal_vec[2]);

        std::vector<double> calc_pos=current_pos;
        calc_pos[0]=calc_pos[0]+0.01;
        goal_vec=goal_pos-calc_pos;
        double goal_x_norm=1/sqrt(goal_vec[0]*goal_vec[0]+goal_vec[1]*goal_vec[1]+goal_vec[2]*goal_vec[2]);

        calc_pos=current_pos;
        calc_pos[1]=calc_pos[1]+0.01;
        goal_vec=goal_pos-calc_pos;
        double goal_y_norm=1/sqrt(goal_vec[0]*goal_vec[0]+goal_vec[1]*goal_vec[1]+goal_vec[2]*goal_vec[2]);

        calc_pos[2]=calc_pos[2]+0.01;
        goal_vec=goal_pos-calc_pos;
        double goal_z_norm=1/sqrt(goal_vec[0]*goal_vec[0]+goal_vec[1]*goal_vec[1]+goal_vec[2]*goal_vec[2]);

        std::vector<double> direc(3);
        direc[0]=goal_x_norm-goal_norm;
        direc[1]=goal_y_norm-goal_norm;
        direc[2]=goal_z_norm-goal_norm;

        return -1*direc;
    }

private:
    double manip_;
    std::vector<double> manip_trans_;
    std::vector<double> fk_mat_;
    std::vector<double> fk_col4_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr manip_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr manip_trans_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr fk_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManipulabilityPlanner>();

    Eigen::Vector3d goal_pos;
    goal_pos<<1,0,1;

    rclcpp::Rate rate(100);
    while (rclcpp::ok()) {
        // 購読トピックのコールバック実行（値を更新）
        rclcpp::spin_some(node);

        // 最新の値を取得
        double manip = node->get_manip();
        std::vector<double> manip_trans = node->get_manip_trans();
        std::vector<double> fk_col4 = node->get_fk_col4();

        Eigen::Vector3d manip_direc;
        Eigen::Vector3d current_pos;

        std::cout << "[manipulability] " << manip << std::endl;
        std::cout << "[manipulability_trans] [";
        for (size_t i = 0; i < manip_trans.size(); ++i) {
            std::cout << manip_trans[i];
            if (i < manip_trans.size() - 1) std::cout << ", ";
            if(i<3){
                manip_direc[i]=manip_trans[i];
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "[fk_matrix 4th column] [";
        for (size_t i = 0; i < fk_col4.size(); ++i) {
            std::cout << fk_col4[i];
            if (i < fk_col4.size() - 1) std::cout << ", ";
            if(i<3){
                current_pos[i]=fk_col4[i];
            }
        }
        std::cout << "]" << std::endl;

        

        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
