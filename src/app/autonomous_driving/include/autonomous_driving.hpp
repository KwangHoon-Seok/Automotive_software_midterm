/**
 * @copyright Hanyang University, Department of Automotive Engineering, 2024. All rights reserved. 
 *            Subject to limited distribution and restricted disclosure only.
 * 
 * @file      autonomous_driving.hpp
 * @brief     autonomous driving algorithm 
 * 
 * @date      2018-11-20 created by Kichun Jo (kichunjo@hanyang.ac.kr)
 *            2023-08-07 updated by Yuseung Na (yuseungna@hanyang.ac.kr)
 *              : adapt new template
 *            2023-08-20 updated by Yuseung Na (yuseungna@hanyang.ac.kr)
 *              : change to ROS2
 *            2024-10-21 updated by Seongjae Jeong (sjeong99@hanyang.ac.kr)
 *              : add the missions
 */

#ifndef __AUTONOMOUS_DRIVING_HPP__
#define __AUTONOMOUS_DRIVING_HPP__
#pragma once

// STD Header
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

// ROS Header
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/message_info.hpp>

// ROS Message Header
#include <ad_msgs/msg/polyfit_lane_data_array.hpp>
#include <ad_msgs/msg/lane_point_data_array.hpp>
#include <ad_msgs/msg/lane_point_data.hpp>
#include <ad_msgs/msg/vehicle_input.hpp>
#include <ad_msgs/msg/vehicle_output.hpp>
#include <ad_msgs/msg/obstacles.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include "visualization_msgs/msg/marker_array.hpp"

// Algorithm Header
#include <eigen3/Eigen/Dense>
#include <fstream>
// autonomous_driving.hpp에서 Point 구조체 정의
struct Point {
    float x, y, z;
    bool visited = false;
    int cluster = -1;

    // 변환 생성자
    Point(const geometry_msgs::msg::Point& msg_point)
        : x(msg_point.x), y(msg_point.y), z(msg_point.z) {}

    // 멤버 초기화 생성자 추가
    Point(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // 기본 생성자와 복사 생성자 유지
    Point() = default;
    Point(const Point&) = default;
    // Equality operator to compare two Points
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Parameter Header

class AutonomousDriving : public rclcpp::Node {
    public:
        AutonomousDriving(const std::string& node_name, const double& loop_rate,
                          const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
        ~AutonomousDriving();

        void Init(const rclcpp::Time& current_time);
        void Run(const rclcpp::Time& current_time);
        void Publish(const rclcpp::Time &current_time, const visualization_msgs::msg::MarkerArray& cluster_markers);
        void UpdateParameter();

    private:
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Functions     
        
        // Callback functions   
        inline void CallbackManualInput(const ad_msgs::msg::VehicleInput::SharedPtr msg) {            
            mutex_manual_input_.lock();
            if(param_use_manual_inputs_ == true) {
                o_vehicle_command_.accel = msg->accel;
                o_vehicle_command_.brake = msg->brake;
                o_vehicle_command_.steering = msg->steering;
            }
            mutex_manual_input_.unlock();
        }
        inline void CallbackVehicleState(const ad_msgs::msg::VehicleOutput::SharedPtr msg) {            
            mutex_vehicle_state_.lock();
            i_vehicle_state_ = *msg;
            mutex_vehicle_state_.unlock();
        }
        inline void CallbackLanePoints(const ad_msgs::msg::LanePointData::SharedPtr msg) {            
            mutex_lane_points_.lock();
            i_lane_points_ = *msg;
            mutex_lane_points_.unlock();
        }
        inline void CallbackObstacles(const ad_msgs::msg::Obstacles::SharedPtr msg) {
            mutex_obstacles_.lock();
            i_obstacles_ = *msg;
            mutex_obstacles_.unlock();
        }
        inline void CallbackLimitSpeed(const std_msgs::msg::Float32::SharedPtr msg) {
            mutex_limit_speed_.lock();
            i_limit_speed_ = msg->data;
            mutex_limit_speed_.unlock();
        }
        
        // Algorithm functions

        
        std::pair<double, double> computeLineModel(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2);
        
        void applyRANSAC(ad_msgs::msg::LanePointData& lane_points, ad_msgs::msg::LanePointData& inliers, int maxIterations = 20, double distanceThreshold = 1.0);
        std::string Vector4dToString(const Eigen::Vector4d& vec);
        Eigen::Vector4d calculateA(const Eigen::VectorXd& X, const Eigen::VectorXd& Y);

        void populatePolyLanes(ad_msgs::msg::PolyfitLaneDataArray& poly_lanes);
        void populateCenterLane(ad_msgs::msg::PolyfitLaneData& driving_way);
        void PurePursuit();
        double computePID(double target_speed, double current_speed, const rclcpp::Time& current_time, double kp);
        double computePID_ACC(double target_distance, double obstacle_distance, const rclcpp::Time& current_time, double kp);
        double getDistanceToVehicle(const ad_msgs::msg::VehicleOutput& vehicle_state, const ad_msgs::msg::VehicleOutput& obstacle);
        // float distance(const Point& a, const Point& b);
        //custom //
        std::vector<int> regionQuery(const std::vector<Point>& points, int pointIdx, float epsilon);
        void expandCluster(std::vector<Point>& points, int pointIdx, int clusterID, float epsilon, size_t min_samples);
        void splitLanePoints(const ad_msgs::msg::LanePointData& lane_points);
        double pointToLineDistance(const geometry_msgs::msg::Point& point, double m, double b);
        // void process_lanes(const ad_msgs::msg::LanePointData& lane_point_LEFT, const ad_msgs::msg::LanePointData& lane_point_RIGHT);
        void process_lanes();
        double pointToCubicDistance(const geometry_msgs::msg::Point& point, double a, double b, double c, double d);
        std::tuple<double, double, double, double> computeCubicModel(
                                                    const geometry_msgs::msg::Point& p1,
                                                    const geometry_msgs::msg::Point& p2,
                                                    const geometry_msgs::msg::Point& p3,
                                                    const geometry_msgs::msg::Point& p4);

        bool isCurveAhead(double curvature_threshold = 0.01);

        visualization_msgs::msg::MarkerArray createClusterMarkers(const std::vector<Point>& points, int cluster_id);

        double calculateCurvature(double x_look_ahead);
        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
        // Variables

        // Publisher
        rclcpp::Publisher<ad_msgs::msg::VehicleInput>::SharedPtr p_vehicle_command_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneData>::SharedPtr p_driving_way_;
        rclcpp::Publisher<ad_msgs::msg::PolyfitLaneDataArray>::SharedPtr p_poly_lanes_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr p_cluster_visualization_;
        // Subscriber
        rclcpp::Subscription<ad_msgs::msg::VehicleInput>::SharedPtr s_manual_input_;
        rclcpp::Subscription<ad_msgs::msg::VehicleOutput>::SharedPtr s_vehicle_state_;
        rclcpp::Subscription<ad_msgs::msg::Obstacles>::SharedPtr s_obstacles_;
        rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr s_limit_speed_;
        rclcpp::Subscription<ad_msgs::msg::LanePointData>::SharedPtr s_lane_points_;
        
        // Timer
        rclcpp::TimerBase::SharedPtr t_run_node_;
        
        // Inputs
        ad_msgs::msg::VehicleOutput i_vehicle_state_;
        ad_msgs::msg::Obstacles i_obstacles_;
        ad_msgs::msg::LanePointData i_lane_points_;
        double i_limit_speed_ = 0.0;

        // Mutex
        std::mutex mutex_manual_input_;
        std::mutex mutex_vehicle_state_;
        std::mutex mutex_obstacles_;
        std::mutex mutex_limit_speed_;
        std::mutex mutex_lane_points_;        

        // Outputs
        ad_msgs::msg::VehicleInput o_vehicle_command_;
        ad_msgs::msg::PolyfitLaneData o_driving_way_;
        ad_msgs::msg::PolyfitLaneDataArray o_poly_lanes_;
        
        double time_prev_csv_lanes_;
        bool param_is_simulator_on_ = false;

        std::string param_vehicle_namespace_;
        bool param_use_manual_inputs_ = false;
        const double param_wheel_base_ = 1.302 + 1.398; // L_f + L_r
        const double param_max_lateral_accel_ = 6200.0 / 1319.91; // Fyf_max / Mass

        // Tuning parameters
        double param_pp_kd_ = 1.0;
        double param_pp_kv_ = 0.0;
        double param_pp_kc_ = 0.0;
        double param_pid_kp_ = 0.0;
        double param_pid_ki_ = 0.01;
        double param_pid_kd_ = 0.0;
        double param_brake_ratio_ = 1.0;

        double param_m_ROIFront_param = 20.0;
        double param_m_ROIRear_param = 10.0;
        double param_m_ROILeft_param = 3.0;
        double param_m_ROIRight_param = 3.0;
        std::string param_ref_csv_path;

        // Algorhtm variables
        double speed_error_integral_ = 0.0;
        double speed_error_prev_     = 0.0;
        double distance_error_integral_ = 0.0;
        double distance_error_prev_     = 0.0;


        // My Variables

        ad_msgs::msg::LanePointData lane_point_LEFT;
        ad_msgs::msg::LanePointData lane_point_RIGHT;
        ad_msgs::msg::LanePointData inliers_LEFT;
        ad_msgs::msg::LanePointData inliers_RIGHT;

        size_t num_points_LEFT;
        size_t num_points_RIGHT;

        size_t number_point;

        Eigen::VectorXd X_LEFT;
        Eigen::VectorXd Y_LEFT;
        Eigen::Vector4d A_LEFT;
        Eigen::VectorXd X_RIGHT;
        Eigen::VectorXd Y_RIGHT;
        Eigen::Vector4d A_RIGHT;
        Eigen::Vector4d A_MID;

        double l_d;
        double g_x = 5.0; // lookahead distance for x axis
        double g_y;
        double e_ld;
        double yaw;
        rclcpp::Time last_time_;
        double max_brake = 1.0;
        double max_accel = 1.0;
        double L = 2.7;
        double closest_obstacle_distance = 0.0;
};

#endif // __AUTONOMOUS_DRIVING_HPP__