/**
 * @file classical_frontier_detection.cpp
 * @author 
 * @brief Implementation of frontier-based exploration
 * @version 0.1
 * @date 2023-03-11
 * 
 * @copyright Copyright (c) 2023
 *
 */
#include "frontier_exploration/classical_frontier_detector.hpp"

FrontierExplorer::FrontierExplorer()
: Node("frontier_explorer")
{
    // Get parameters
    region_size_thresh_ = this->declare_parameter("region_size_thresh", 4);
    robot_width_ = this->declare_parameter("robot_width", 0.5);
    occupancy_map_topic_ = this->declare_parameter("occupancy_map_msg", "map");

    // Subscribers/Publichers/Service setup
    map_subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
        occupancy_map_topic_, 1, std::bind(&FrontierExplorer::map_callback, this, _1));

    service_ = this->create_service<frontier_interfaces::srv::FrontierGoal>(
        "frontier_pose", std::bind(&FrontierExplorer::get_frontiers, this, _1, _2));

    marker_publisher_ = this->create_publisher<visualization_msgs::msg::Marker>("f_markers", 1);
    frontier_map_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("f_map", 1);

    //tf listner 
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

void FrontierExplorer::map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr recent_map)
{
    // auto width = recent_map->info.width;
    // auto height = recent_map->info.height;
    // RCLCPP_INFO(this->get_logger(),"Map recieved w: %d h: %d.", width, height);

    std::lock_guard<std::mutex> guard(mutex_);
    map_ = *recent_map;

    // Return early if the map data is empty
    if (map_.data.empty()) return;

    // Compute frontiers and clusters in real-time upon receiving map
    std::vector<cell> processed = preprocessMap(map_.data, map_.info.width, map_.info.height, 5);

    frontierCellGrid_ = computeFrontierCellGrid(processed, map_.info.width, map_.info.height);
    frontierRegions_ = computeFrontierRegions(frontierCellGrid_, map_.info.width, map_.info.height, 
        map_.info.resolution, map_.info.origin.position.x, map_.info.origin.position.y, region_size_thresh_);

    // Publish f_map and visualization markers in real-time
    nav_msgs::msg::OccupancyGrid f_map = map_;
    f_map.data = processed;
    frontier_map_publisher_->publish(f_map);

    publishFrontiers();
}

void FrontierExplorer::get_frontiers(const std::shared_ptr<frontier_interfaces::srv::FrontierGoal::Request> request,
          std::shared_ptr<frontier_interfaces::srv::FrontierGoal::Response> response)
{
    RCLCPP_INFO(this->get_logger(), "Received request, %d", request->goal_rank);

    // Copy the map
    std::unique_lock<std::mutex> lck(mutex_);
    std::vector<frontierRegion> regions = frontierRegions_;
    lck.unlock();

    // // Pre-process the grid cell map
    // std::vector<cell> processed = preprocessMap(map.data, map.info.width, map.info.height, 5);

    // // Compute frontier grid cell map
    // frontierCellGrid_.clear();
	// frontierCellGrid_ = computeFrontierCellGrid(processed, map.info.width, map.info.height);
	
	// // Compute the Frontier Regions
	// frontierRegions_.clear();
	// frontierRegions_ = computeFrontierRegions(frontierCellGrid_, map.info.width, map.info.height, 
    //     map.info.resolution, map.info.origin.position.x, map.info.origin.position.y, region_size_thresh_);

    // nav_msgs::msg::OccupancyGrid f_map = map;

    // // std::transform(frontierCellGrid_.begin(), frontierCellGrid_.end(), frontierCellGrid_.begin(),
    // //            std::bind(std::multiplies<cell>(), std::placeholders::_1, 255));
    // f_map.data = processed;
    // frontier_map_publisher_->publish(f_map);

    // publishFrontiers();

    // Get robot position
    geometry_msgs::msg::TransformStamped stransform;
    double rx = 0.0, ry = 0.0;
    try{
        stransform = tf_buffer_->lookupTransform(map_frame_, base_frame_,
                                                    tf2::TimePointZero, tf2::durationFromSec(1.0));
        rx = stransform.transform.translation.x;
        ry = stransform.transform.translation.y;
    }
    catch (const tf2::TransformException &ex){
        RCLCPP_WARN(this->get_logger(), "TF lookup failed in get_frontiers: %s", ex.what());
    }
    
    // Find best goal based on position and size
    frontierRegion goal = selectFrontier(regions, request->goal_rank, rx, ry);

    double gx = goal.x;
    double gy = goal.y;

    // Apply goal pullback only when a valid frontier exists (size > 0)
    if (goal.size > 0) {
        double dist_to_robot = std::hypot(gx - rx, gy - ry);
        if (dist_to_robot > 0.8) {
            double pullback = 0.3; // Pull back 0.3m toward the robot (into open space)
            gx = gx - (pullback / dist_to_robot) * (gx - rx);
            gy = gy - (pullback / dist_to_robot) * (gy - ry);
        }
    } else {
        gx = 0.0;
        gy = 0.0;
    }

    // Create and init message
    geometry_msgs::msg::PoseStamped goal_pose;
    goal_pose.header.stamp = this->get_clock()->now();
    // An empty frame_id tells the client that no valid frontier exists for this rank.
    // (Using (0,0) as a marker breaks when a real frontier lies at the map origin,
    // which is the robot start pose in SLAM/Habitat.)
    goal_pose.header.frame_id = (goal.size > 0) ? map_frame_ : "";
    goal_pose.pose.position.x = gx; // Assign the computed safe position (gx)
    goal_pose.pose.position.y = gy; // Assign the computed safe position (gy)
    goal_pose.pose.position.z = 0.0;
    // Set valid default quaternion orientation (w = 1.0)
    goal_pose.pose.orientation.x = 0.0;
    goal_pose.pose.orientation.y = 0.0;
    goal_pose.pose.orientation.z = 0.0;
    goal_pose.pose.orientation.w = 1.0;
    
    // Set the response
    response->goal_pose = goal_pose;

    RCLCPP_INFO(this->get_logger(), "Sending goal x: %f y: %f.",
        goal_pose.pose.position.x, goal_pose.pose.position.y);
}

void FrontierExplorer::publishFrontiers()
{
    // -------------------------------------------------------------
    // 1. [Green Marker] Detected frontier edge cells (small points)
    // -------------------------------------------------------------
    visualization_msgs::msg::Marker all_points;
    all_points.header.frame_id = map_frame_;
    all_points.header.stamp = this->get_clock()->now();
    all_points.ns = "all_frontiers";
    all_points.id = 0; // Unique marker ID
    all_points.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    all_points.action = visualization_msgs::msg::Marker::ADD;
    
    // Point size: 0.05m
    all_points.scale.x = 0.05;
    all_points.scale.y = 0.05;
    all_points.scale.z = 0.05;
    
    // Color: Green
    all_points.color.r = 0.0f;
    all_points.color.g = 1.0f;
    all_points.color.b = 0.2f;
    all_points.color.a = 1.0f;

    for (uint32_t i = 0; i < map_.info.height; ++i) {
        for (uint32_t j = 0; j < map_.info.width; ++j) {
            if (frontierCellGrid_[i * map_.info.width + j] == 1) {
                geometry_msgs::msg::Point p;
                p.x = map_.info.origin.position.x + (j + 0.5) * map_.info.resolution;
                p.y = map_.info.origin.position.y + (i + 0.5) * map_.info.resolution;
                p.z = 0.03; // Slightly above ground level
                all_points.points.push_back(p);
            }
        }
    }
    // Publish green frontier point markers
    marker_publisher_->publish(all_points);


    // -------------------------------------------------------------
    // 2. [Red Marker] Clustered frontier centroid points (large points)
    // -------------------------------------------------------------
    visualization_msgs::msg::Marker cluster_centroids;
    cluster_centroids.header.frame_id = map_frame_;
    cluster_centroids.header.stamp = this->get_clock()->now();
    cluster_centroids.ns = "cluster_centroids";
    cluster_centroids.id = 1; // Unique ID to prevent overwriting other markers
    cluster_centroids.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    cluster_centroids.action = visualization_msgs::msg::Marker::ADD;
    
    // Point size: 0.1m
    cluster_centroids.scale.x = 0.1;
    cluster_centroids.scale.y = 0.1;
    cluster_centroids.scale.z = 0.1;
    
    // Color: Red
    cluster_centroids.color.r = 1.0f;
    cluster_centroids.color.g = 0.0f;
    cluster_centroids.color.b = 0.0f;
    cluster_centroids.color.a = 1.0f;

    for (const auto & reg : frontierRegions_) {
        geometry_msgs::msg::Point p;
        p.x = reg.x;
        p.y = reg.y;
        p.z = 0.08; // Placed above the green points for visibility
        cluster_centroids.points.push_back(p);
    }
    // Publish red centroid markers
    marker_publisher_->publish(cluster_centroids);
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    
    // Start processing data from the node as well as the callbacks and the timer
    rclcpp::spin(std::make_shared<FrontierExplorer>());
    
    // Shutdown the node when finished
    rclcpp::shutdown();
    return 0;
}