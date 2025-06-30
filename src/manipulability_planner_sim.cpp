#include <memory>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "eigen3/Eigen/Dense"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <limits>
#include <thread>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>

#include "motion_planning/nanoflann.hpp"

class ManipulabilityPlanner : public rclcpp::Node
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

    ManipulabilityPlanner()
    : Node("manipulability_planner"),
      manip_(0.0),
      manip_trans_(Eigen::VectorXd::Zero(6)),
      fk_mat_(Eigen::Matrix4d::Zero()),
      fk_col4_(Eigen::Vector4d::Zero()),
      got_pointcloud_(false),
      got_marker1_(false),
      got_marker2_(false),
      kd_tree_built_(false)
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
            });

        fk_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
            "fk_matrix", 10,
            [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
                if (msg->data.size() == 16) {
                    fk_mat_ = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(msg->data.data());
                    fk_col4_ = fk_mat_.col(3);
                }
            });
        
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

        // 停止合図のパブリッシャ
        pointcloud_acquired_pub_ = this->create_publisher<std_msgs::msg::Bool>("/pointcloud_acquired", 1);
        destination_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("destination", 10);
        execution_pub_ = this->create_publisher<std_msgs::msg::Bool>("/execution", 1);
        movement_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("movement", 10);

        check_sub_ = this->create_subscription<std_msgs::msg::Float64>(
            "check", 10,
            [this](const std_msgs::msg::Float64::SharedPtr msg) {
                check_ = msg->data;
            });
        
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

        
    }

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr destination_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr execution_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr movement_pub_;

    // PointCloud2→Eigen行列（Nx3）
    // void parse_pointcloud(const sensor_msgs::msg::PointCloud2 & msg) {
    //     size_t n = msg.width * msg.height;
    //     saved_cloud_mat_.resize(n, 3);
    //     sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg, "x");
    //     sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg, "y");
    //     sensor_msgs::PointCloud2ConstIterator<float> iter_z(msg, "z");
    //     for (size_t i = 0; i < n; ++i, ++iter_x, ++iter_y, ++iter_z) {
    //         saved_cloud_mat_(i, 0) = *iter_x;
    //         saved_cloud_mat_(i, 1) = *iter_y;
    //         saved_cloud_mat_(i, 2) = *iter_z;
    //     }
    // }

    // 最近傍点を抽出
    Eigen::Vector3d get_nearest_point(const Eigen::Vector3d& next_pos) const {
        if (!got_pointcloud_) return Eigen::Vector3d::Zero();
        Eigen::Vector3f next_pos_f = next_pos.cast<float>();
        Eigen::MatrixXf diff = saved_cloud_mat_.rowwise() - next_pos_f.transpose();
        Eigen::VectorXf dists = diff.rowwise().norm();
        Eigen::Index minIndex;
        float minDist = dists.minCoeff(&minIndex);
        // float型（行列）→ double型（ベクトル）にキャストして返す
        return saved_cloud_mat_.row(minIndex).cast<double>();
    }

    Eigen::Vector3d get_nearest_point2(const Eigen::Vector3d& next_pos, double roi = 0.05) const {
        if (!got_pointcloud_) return Eigen::Vector3d::Zero();

        std::vector<int> candidates;
        for (int i = 0; i < saved_cloud_mat_.rows(); ++i) {
            if ( (saved_cloud_mat_(i,0) > next_pos.x() - roi) && (saved_cloud_mat_(i,0) < next_pos.x() + roi) &&
                (saved_cloud_mat_(i,1) > next_pos.y() - roi) && (saved_cloud_mat_(i,1) < next_pos.y() + roi) &&
                (saved_cloud_mat_(i,2) > next_pos.z() - roi) && (saved_cloud_mat_(i,2) < next_pos.z() + roi) )
            {
                candidates.push_back(i);
            }
        }
        // ROI内に点がなければ従来通り全点探索
        if (candidates.empty()) {
            return get_nearest_point(next_pos); // 既存の全点探索メソッド
        }

        int min_idx = -1;
        double min_dist = std::numeric_limits<double>::max();
        for (int idx : candidates) {
            double dist = (saved_cloud_mat_.row(idx).cast<double>() - next_pos.transpose()).norm();
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = idx;
            }
        }
        if (min_idx >= 0) {
            return saved_cloud_mat_.row(min_idx).cast<double>();
        }
        return Eigen::Vector3d::Zero();
    }


    double get_manip() const { return manip_; }
    Eigen::VectorXd get_manip_trans() const { return manip_trans_; }
    Eigen::Vector4d get_fk_col4() const { return fk_col4_; }
    Eigen::Matrix4d get_fk_mat() const { return fk_mat_; }

    bool got_pointcloud() const { return got_pointcloud_; }
    const sensor_msgs::msg::PointCloud2 & get_pointcloud() const { return saved_cloud_; }
    bool got_marker1() const { return got_marker1_; }
    bool got_marker2() const { return got_marker2_; }
    Eigen::Vector3d get_marker1() const { return marker1_pos_; }
    Eigen::Vector3d get_marker2() const { return marker2_pos_; }

    double get_check() const { return check_; }
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
    std::vector<std::pair<size_t, float>> ret_matches;
    nanoflann::SearchParams params;
    kd_tree_->radiusSearch(query_pt, search_radius, ret_matches, params);

    if (ret_matches.empty()) {
        // ROI内に点がなければグローバル最近傍
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        kd_tree_->findNeighbors(resultSet, query_pt, nanoflann::SearchParams(10));
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


private:
    double manip_;
    Eigen::VectorXd manip_trans_;  // size 6
    Eigen::Matrix4d fk_mat_;       // 4x4
    Eigen::Vector4d fk_col4_;      // 4x1
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr manip_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr manip_trans_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr fk_sub_;

    Eigen::MatrixXf saved_cloud_mat_; // Nx3
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
    auto node = std::make_shared<ManipulabilityPlanner>();
    rclcpp::Rate rate(10);
    std_msgs::msg::Bool execution_msg;
    execution_msg.data=false;
    int count=0;
    Eigen::Vector3d start_pos;
    start_pos<<0.02,0.393,0.8;
    Eigen::Vector3d goal_pos;
    goal_pos<<0.42,0.393,0.8;

    geometry_msgs::msg::WrenchStamped force_torque;

    // 点群が来るまで待機
    while (rclcpp::ok() && (!node->got_pointcloud() || !node->got_marker1() || !node->got_marker2())) {
        rclcpp::spin_some(node);
        node->execution_pub_->publish(execution_msg);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if(count==0) std::cout<<"waiting for pointcloud "<<std::endl;
        force_torque=node->get_force_torque();
        std::cout<<force_torque.wrench.force.x<<std::endl;
        count++;
    }
    start_pos=node->get_marker1();
    goal_pos=node->get_marker2();

    //点群なしシミュレーション
    // while (rclcpp::ok() && (!node->get_manip())) {
    //     rclcpp::spin_some(node);
    //     node->execution_pub_->publish(execution_msg);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(20));
    //     if(count==0) std::cout<<"waiting for publish "<<std::endl;
    //     count++;
    // }

    count=0;

    Eigen::Vector4d fk_col4 = node->get_fk_col4();
    Eigen::Vector3d current_pos = fk_col4.head<3>();
    Eigen::VectorXd manip_trans = node->get_manip_trans();

    execution_msg.data=true;
    
    //開始位置に到達するまで待機
    while(rclcpp::ok()){
        rclcpp::spin_some(node);

        fk_col4 = node->get_fk_col4();
        current_pos = fk_col4.head<3>();
        std_msgs::msg::Float64MultiArray destination_msg;
        destination_msg.data.resize(start_pos.size());
        for (int i = 0; i < start_pos.size(); ++i) {
            destination_msg.data[i] = start_pos[i];
        }
        node->destination_pub_->publish(destination_msg);

        Eigen::Vector3d dist_vec=start_pos-current_pos;
        double dist=dist_vec.norm();
        if(dist<0.005){
            std::cout<<"start   x: "<<current_pos(0)<<"  y: "<<current_pos(1)<<"  z: "<<current_pos(2)<<std::endl;
            //開始位置に到達
            execution_msg.data=false;
            node->execution_pub_->publish(execution_msg);
            break;
        }
        if(count==0) std::cout<<"waiting for reach "<<std::endl;
        node->execution_pub_->publish(execution_msg);
        count++;
        rate.sleep();
    }
    count=0;

    double coef_manip=0.5;
    double coef_pos=1.0;
    
    while (rclcpp::ok()) {
        rclcpp::spin_some(node);
        node->execution_pub_->publish(execution_msg);
        if(count>100) break;
        count++;
    }
    count=0;

    //追従制御開始
    while (rclcpp::ok()) {
        std::chrono::system_clock::time_point start, end;
        start = std::chrono::system_clock::now();

        rclcpp::spin_some(node);

        double manip = node->get_manip();
        manip_trans = node->get_manip_trans();
        fk_col4 = node->get_fk_col4();

        Eigen::Vector3d manip_direc = manip_trans.head<3>();
        current_pos = fk_col4.head<3>();
        Eigen::Vector3d goal_vec=goal_pos-current_pos;
        double distance=goal_vec.norm();
        // std::cout<<"x pos "<<current_pos(0)<<std::endl;
        // std::cout<<"distance "<<distance<<std::endl;
        std::cout<<"current:    "<<current_pos.transpose()<<std::endl;

        Eigen::Vector3d goal_pot=node->potential(current_pos, goal_pos,distance);
        Eigen::Vector3d direction=coef_manip*manip_direc+coef_pos*goal_pot;

        // Eigen::Vector3d next_pos=current_pos+direction;
        double stepsize=0.003;
        Eigen::Vector3d move_trans=stepsize*direction/direction.norm();
        // std::cout<<"move_trans: "<<move_trans.transpose()<<std::endl;
        Eigen::Vector3d next_pos=current_pos+move_trans;
        // std::cout<<"next:      "<<next_pos.transpose()<<std::endl;

        // Eigen::Vector3d nearest_point = node->get_nearest_point(next_pos);
        // Eigen::Vector3d nearest_point = node->get_nearest_point2(next_pos,0.05);
        Eigen::Vector3d nearest_point = node->get_nearest_point3(next_pos,0.05);
        // std::cout << "nearest:    " << nearest_point.transpose() << std::endl;

        // Eigen::Vector3d nearest_point = node->get_nearest_point(next_pos);
        // Eigen::Vector3d nearest_point = node->get_nearest_point2(next_pos,0.05);
        // std::cout << "nearest:    " << nearest_point.transpose() << std::endl;

        Eigen::Vector3d nearest_move=nearest_point-current_pos;
        // std::cout<<"near_move:  "<<nearest_move.transpose()<<std::endl;

        Eigen::Matrix<double, 6,1> movement;
        movement<<nearest_move,0,0,0;
        // movement<<0.01*direction/direction.norm(),0,0,0;

        std_msgs::msg::Float64MultiArray movement_msg;
        movement_msg.data.resize(6);
        for (int i = 0; i < 6; ++i) {
            movement_msg.data[i] = movement[i];
        }
        if (distance<0.01){
            for (int i = 0; i < 6; ++i) {
                movement_msg.data[i] = 0;
            }
            node->movement_pub_->publish(movement_msg);
            execution_msg.data=true;
            node->execution_pub_->publish(execution_msg);
            std::cout<<"end     x: "<<current_pos(0)<<"  y: "<<current_pos(1)<<"  z: "<<current_pos(2)<<std::endl;
            std::cout<<"goal: "<<goal_pos.transpose()<<std::endl;
            break;
        }
        node->movement_pub_->publish(movement_msg);
        // std::cout << "count: " << count << std::endl;
        // std::cout << "check: " << node->get_check() << std::endl;
        count++;
        //std::cout<<"movement "<<movement<<std::endl;

        end = std::chrono::system_clock::now();
        double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
        printf("time %lf[ms]\n", time);

        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
