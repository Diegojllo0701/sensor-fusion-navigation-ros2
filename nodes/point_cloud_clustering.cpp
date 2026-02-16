#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <Eigen/Dense>

struct DetectedPlane
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr points;
    Eigen::Vector3f normal;
    enum Type { FLOOR, CEILING, WALL, UNKNOWN } type;
};

class PointCloudClusteringNode : public rclcpp::Node 
{
public:
    PointCloudClusteringNode() : Node("point_cloud_clustering")
    {
        RCLCPP_INFO(this->get_logger(), "Node started, listening to: /camera/points");
        subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/points",
            10,
            std::bind(&PointCloudClusteringNode::callback, this, std::placeholders::_1)
        );
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "filtered_points",
            10
        );
    }   
private:
    void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) return;
        auto depthfiltered = filterDepth(cloud);
        auto downsampled = downsample(depthfiltered);
        auto planes = removeFloorPlanes(downsampled);
        

        RCLCPP_INFO(this->get_logger(), ("Size: " + std::to_string(downsampled->size())+ ", Original Size: " + std::to_string(cloud->size())).c_str());

        sensor_msgs::msg::PointCloud2 outputmsg;
        pcl::toROSMsg(*planes,outputmsg);
        outputmsg.header = msg->header;
        publisher_->publish(outputmsg);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filterDepth(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        
        
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.1, 7.9);
        pass.filter(*filtered);
        
        return filtered;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.05, 0.05, 0.05);
        vg.filter(*cloud_filtered);
        
        return cloud_filtered;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr removeFloorPlanes(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        auto remaining = cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr non_floor(new pcl::PointCloud<pcl::PointXYZ>);
        int min_plane_points = 200;

        while (remaining->size() > min_plane_points)
        {
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(100);
            seg.setDistanceThreshold(0.02);
            seg.setInputCloud(remaining);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.size() < min_plane_points) break;

            float a = coefficients->values[0];
            float b = coefficients->values[1];
            float c = coefficients->values[2];

            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(remaining);
            extract.setIndices(inliers);

            pcl::PointCloud<pcl::PointXYZ>::Ptr plane_points(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setNegative(false);
            extract.filter(*plane_points);

            pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
            extract.setNegative(true);
            extract.filter(*temp);
            remaining = temp; 

            if (std::abs(b) > 0.9)
            {
                RCLCPP_INFO(this->get_logger(), 
                    "Discarding floor, normal: [%.2f, %.2f, %.2f]", a, b, c);
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), 
                    "Keeping wall, normal: [%.2f, %.2f, %.2f]", a, b, c);
                *non_floor += *plane_points; 
            }
        }

        *non_floor += *remaining;

        return non_floor;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudClusteringNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}