#include <memory>
#include <vector>
#include <iostream>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/filters/extract_indices.h>
#include <Eigen/Core>

#include <pcl/visualization/pcl_visualizer.h>

using namespace std::chrono_literals;

class NormalClusterNode : public rclcpp::Node
{
public:
    NormalClusterNode()
        : Node("normal_cluster_node"), received_point_cloud_(false),viewer_(new pcl::visualization::PCLVisualizer("Normals & Clusters"))
    {
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/capsule_cloud_transformed", 1,
            std::bind(&NormalClusterNode::cloud_callback, this, std::placeholders::_1));
        timer_ = this->create_wall_timer(1s, std::bind(&NormalClusterNode::timer_callback, this));
        viewer_->setBackgroundColor(1.0, 1.0, 1.0);
        viewer_->addCoordinateSystem(0.5);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    bool received_point_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_{new pcl::PointCloud<pcl::PointXYZ>};
    pcl::PointCloud<pcl::PointNormal>::Ptr normals_{new pcl::PointCloud<pcl::PointNormal>};
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> clusters_;
    pcl::visualization::PCLVisualizer::Ptr viewer_;

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        if (received_point_cloud_) return; // 1回のみ取得
        RCLCPP_INFO(this->get_logger(), "PointCloud received.");
        pcl::fromROSMsg(*msg, *cloud_);
        normals_->clear();
        pcl::copyPointCloud(*cloud_, *normals_);
        estimate_normals();
        clustering();
        output_cluster_normals();
        received_point_cloud_ = true;
    }

    void timer_callback()
    {
        if (received_point_cloud_) {
            // 処理済みクラスタ法線方向を表示するだけ
            output_cluster_normals();
        }
        viewer_->spinOnce(100);
    }

    void estimate_normals()
    {
        pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> ne;
        ne.setInputCloud(normals_);
        auto tree = pcl::search::KdTree<pcl::PointNormal>::Ptr(new pcl::search::KdTree<pcl::PointNormal>);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.5);
        ne.compute(*normals_);
        RCLCPP_INFO(this->get_logger(), "Normals estimated.");
    }

    // クラスタリング
    void clustering()
    {
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::ConditionalEuclideanClustering<pcl::PointNormal> cec(true);
        cec.setInputCloud(normals_);
        cec.setConditionFunction(&NormalClusterNode::custom_condition);
        cec.setClusterTolerance(0.1);
        cec.setMinClusterSize(100);
        cec.setMaxClusterSize(normals_->points.size());
        cec.segment(cluster_indices);

        clusters_.clear();
        pcl::ExtractIndices<pcl::PointNormal> ei;
        ei.setInputCloud(normals_);
        ei.setNegative(false);
        for(const auto& indices : cluster_indices){
            pcl::PointCloud<pcl::PointNormal>::Ptr tmp_cluster(new pcl::PointCloud<pcl::PointNormal>);
            pcl::PointIndices::Ptr indices_ptr(new pcl::PointIndices(indices));
            ei.setIndices(indices_ptr);
            ei.filter(*tmp_cluster);
            clusters_.push_back(tmp_cluster);
        }
        RCLCPP_INFO(this->get_logger(), "Clustering done. #clusters: %zu", clusters_.size());
    }

    // クラスタごとの平均法線を出力
    // void output_cluster_normals()
    // {
    //     for(size_t i=0; i<clusters_.size(); ++i){
    //         Eigen::Vector3d avg_normal(0,0,0);
    //         int count = 0;
    //         for(const auto& pt : clusters_[i]->points){
    //             Eigen::Vector3d n(pt.normal_x, pt.normal_y, pt.normal_z);
    //             if(n.norm() > 1e-3) {
    //                 avg_normal += n;
    //                 count++;
    //             }
    //         }
    //         if(count > 0) avg_normal /= count;
    //         std::cout << "Cluster " << i
    //                   << ": normal = (" << avg_normal.x() << ", " << avg_normal.y() << ", " << avg_normal.z() << ")"
    //                   << ", points = " << count << std::endl;
    //     }
    // }

    void output_cluster_normals()
    {
        viewer_->removeAllPointClouds();
        viewer_->removeAllShapes();

        // オリジナル点群描画
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> orig_color(cloud_, 100, 100, 100);
        viewer_->addPointCloud(cloud_, orig_color, "cloud");
        viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

        // クラスタを色分け描画＋法線ベクトルも表示
        for(size_t i=0; i<clusters_.size(); ++i){
            std::string name = "cluster_" + std::to_string(i);
            int r = (i*77)%256, g = (i*130)%256, b = (i*200)%256; // 適当な色分け

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> cluster_color(clusters_[i], r, g, b);
            viewer_->addPointCloud<pcl::PointNormal>(clusters_[i], cluster_color, name);
            viewer_->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);

            // 法線ベクトルを数点ごとに描画
            viewer_->addPointCloudNormals<pcl::PointNormal>(clusters_[i], 10, 0.2, name + "_normals");

            // 平均法線も出力
            Eigen::Vector3d avg_normal(0,0,0);
            int count = 0;
            for(const auto& pt : clusters_[i]->points){
                Eigen::Vector3d n(pt.normal_x, pt.normal_y, pt.normal_z);
                if(n.norm() > 1e-3) {
                    avg_normal += n;
                    count++;
                }
            }
            if(count > 0) avg_normal /= count;
            std::cout << "Cluster " << i
                      << ": normal = (" << avg_normal.x() << ", " << avg_normal.y() << ", " << avg_normal.z() << ")"
                      << ", points = " << count << std::endl;
        }
    }

    // 法線方向の差が閾値未満なら同一クラスタ
    static bool custom_condition(const pcl::PointNormal& seed, const pcl::PointNormal& candidate, float)
    {
        Eigen::Vector3d n1(seed.normal_x, seed.normal_y, seed.normal_z);
        Eigen::Vector3d n2(candidate.normal_x, candidate.normal_y, candidate.normal_z);
        if(n1.norm() < 1e-3 || n2.norm() < 1e-3) return false;
        double angle = std::acos(n1.dot(n2)/(n1.norm()*n2.norm()));
        constexpr double threshold_deg = 1.0;
        return angle/M_PI*180.0 < threshold_deg;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NormalClusterNode>());
    rclcpp::shutdown();
    return 0;
}
