#include <memory>
#include <vector>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

class ManipulabilityPlanner : public rclcpp::Node
{
public:
    ManipulabilityPlanner()
    : Node("manipulability_planner"), manip_(0.0), manip_trans_(6, 0.0)
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
    }

    double get_manip() const { return manip_; }
    std::vector<double> get_manip_trans() const { return manip_trans_; }

private:
    double manip_;
    std::vector<double> manip_trans_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr manip_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr manip_trans_sub_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ManipulabilityPlanner>();

    rclcpp::Rate rate(100); // 10Hz
    while (rclcpp::ok()) {
        // 購読トピックのコールバック実行（値を更新）
        rclcpp::spin_some(node);

        // 最新の値を取得
        double manip = node->get_manip();
        std::vector<double> manip_trans = node->get_manip_trans();

        std::cout << "[manipulability] " << manip << std::endl;
        std::cout << "[manipulability_trans] [";
        for (size_t i = 0; i < manip_trans.size(); ++i) {
            std::cout << manip_trans[i];
            if (i < manip_trans.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
