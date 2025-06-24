#include <memory>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "eigen3/Eigen/Dense"

class ManipulabilityPlanner : public rclcpp::Node
{
public:
    ManipulabilityPlanner()
    : Node("manipulability_planner"),
      manip_(0.0),
      manip_trans_(Eigen::VectorXd::Zero(6)),
      fk_mat_(Eigen::Matrix4d::Zero()),
      fk_col4_(Eigen::Vector4d::Zero())
    {
        manip_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "manipulability", 10,
            [this](const std_msgs::msg::Float64::SharedPtr msg) {
                manip_ = msg->data;
            });

        manip_trans_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "manipulability_trans", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() == 6)
                    manip_trans_ = Eigen::Map<const Eigen::VectorXd>(msg->data.data(), 6);
                // 必要ならelseでリサイズ/0埋めも考慮
            });

        fk_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "fk_matrix", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() == 16) {
                    fk_mat_ = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(msg->data.data());
                    fk_col4_ = fk_mat_.col(3);
                }
            });
    }

    double get_manip() const { return manip_; }
    Eigen::VectorXd get_manip_trans() const { return manip_trans_; }
    Eigen::Vector4d get_fk_col4() const { return fk_col4_; }
    Eigen::Matrix4d get_fk_mat() const { return fk_mat_; }

    // Eigen化したpotential
    Eigen::Vector3d potential(const Eigen::Vector3d& current_pos, const Eigen::Vector3d& goal_pos) {
        Eigen::Vector3d goal_vec = goal_pos - current_pos;
        double goal_norm = 1.0 / goal_vec.norm();

        Eigen::Vector3d calc_pos = current_pos;
        calc_pos[0] += 0.01;
        goal_vec = goal_pos - calc_pos;
        double goal_x_norm = 1.0 / goal_vec.norm();

        calc_pos = current_pos;
        calc_pos[1] += 0.01;
        goal_vec = goal_pos - calc_pos;
        double goal_y_norm = 1.0 / goal_vec.norm();

        calc_pos = current_pos;
        calc_pos[2] += 0.01;
        goal_vec = goal_pos - calc_pos;
        double goal_z_norm = 1.0 / goal_vec.norm();

        Eigen::Vector3d direc;
        direc[0] = goal_x_norm - goal_norm;
        direc[1] = goal_y_norm - goal_norm;
        direc[2] = goal_z_norm - goal_norm;

        return -direc;
    }

private:
    double manip_;
    Eigen::VectorXd manip_trans_;  // size 6
    Eigen::Matrix4d fk_mat_;       // 4x4
    Eigen::Vector4d fk_col4_;      // 4x1
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr manip_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr manip_trans_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr fk_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManipulabilityPlanner>();

    Eigen::Vector3d goal_pos;
    goal_pos << 1, 0, 1;

    rclcpp::Rate rate(100);
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);

        double manip = node->get_manip();
        Eigen::VectorXd manip_trans = node->get_manip_trans();
        Eigen::Vector4d fk_col4 = node->get_fk_col4();

        Eigen::Vector3d manip_direc = manip_trans.head<3>();
        Eigen::Vector3d current_pos = fk_col4.head<3>();

        std::cout << "[manipulability] " << manip << std::endl;

        std::cout << "[manipulability_trans] [";
        for (int i = 0; i < manip_trans.size(); ++i) {
            std::cout << manip_trans[i];
            if (i < manip_trans.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "[fk_matrix 4th column] [";
        for (int i = 0; i < fk_col4.size(); ++i) {
            std::cout << fk_col4[i];
            if (i < fk_col4.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // potentialもEigen版で呼び出せる
        Eigen::Vector3d pvec = node->potential(current_pos, goal_pos);
        std::cout << "[potential direction] [" << pvec.transpose() << "]" << std::endl;

        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
