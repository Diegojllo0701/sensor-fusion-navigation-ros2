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
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h> 
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>

struct DetectedPlane
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr points;
    Eigen::Vector3f normal;
    enum Type { FLOOR, CEILING, WALL, UNKNOWN } type;
};

struct Obstacle
{
    Eigen::Vector3f centroid;      
    Eigen::Vector3f min_point;     
    Eigen::Vector3f max_point;     
    Eigen::Vector3f dimensions;    
    pcl::PointCloud<pcl::PointXYZ>::Ptr points;
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
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "obstacle_markers", 10
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
        auto without_floor = removeFloorPlanes(downsampled);
        auto obstacles = clusterObstacles(without_floor);
        
        publishObstacleMarkers(obstacles, msg->header);

        sensor_msgs::msg::PointCloud2 outputmsg;
        pcl::toROSMsg(*without_floor,outputmsg);
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

    std::vector<Obstacle> clusterObstacles(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        std::vector<Obstacle> obstacles;

        if (cloud->empty()) return obstacles;

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.1); 
        ec.setMinClusterSize(30);
        ec.setMaxClusterSize(5000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        for (const auto& indices : cluster_indices)
        {
            Obstacle obs;
            obs.points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            
            for (const auto& idx : indices.indices)
            {
                obs.points->push_back((*cloud)[idx]);
            }

            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*obs.points, min_pt, max_pt);
            
            obs.min_point = Eigen::Vector3f(min_pt.x, min_pt.y, min_pt.z);
            obs.max_point = Eigen::Vector3f(max_pt.x, max_pt.y, max_pt.z);
            
            obs.dimensions = obs.max_point - obs.min_point;
            
            obs.centroid = (obs.min_point + obs.max_point) / 2.0f;
            
            obstacles.push_back(obs);
            
            RCLCPP_INFO(this->get_logger(), 
                "Obstacle at [%.2f, %.2f, %.2f], size [%.2f x %.2f x %.2f]m",
                obs.centroid.x(), obs.centroid.y(), obs.centroid.z(),
                obs.dimensions.x(), obs.dimensions.y(), obs.dimensions.z());
        }

        return obstacles;
    }

    void publishObstacleMarkers(const std::vector<Obstacle>& obstacles, const std_msgs::msg::Header& header)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        
        for (size_t i = 0; i < obstacles.size(); i++)
        {
            const auto& obs = obstacles[i];
            
            visualization_msgs::msg::Marker marker;
            marker.header = header;
            marker.ns = "obstacles";
            marker.id = i;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            
            // Position (center of bounding box)
            marker.pose.position.x = obs.centroid.x();
            marker.pose.position.y = obs.centroid.y();
            marker.pose.position.z = obs.centroid.z();
            marker.pose.orientation.w = 1.0;
            
            // Size (bounding box dimensions)
            marker.scale.x = obs.dimensions.x();
            marker.scale.y = obs.dimensions.y();
            marker.scale.z = obs.dimensions.z();
            
            // Color (semi-transparent red)
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.color.a = 0.5;
            
            marker.lifetime = rclcpp::Duration::from_seconds(0.5);
            
            marker_array.markers.push_back(marker);
        }
        
        marker_publisher_->publish(marker_array);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudClusteringNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}