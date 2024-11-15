/**
 * @copyright Hanyang University, Department of Automotive Engineering, 2024. All rights reserved. 
 *            Subject to limited distribution and restricted disclosure only.
 *            
 * @file      evaluation_node.cpp
 * @brief     autonomous driving algorithm evaluation tool
 * 
 * @date      2018-11-28 created by Eunsan Jo (eunsan.mountain@gmail.com)
 *            2023-08-07 updated by Yuseung Na (yuseungna@hanyang.ac.kr)
 *              : adapt new template
 *            2023-08-20 updated by Yuseung Na (yuseungna@hanyang.ac.kr)
 *              : change to ROS2
 */

#include "evaluation_node.hpp"

Evaluation::Evaluation(const std::string& node_name, const double& loop_rate,
                       const rclcpp::NodeOptions& options)
    : Node(node_name, options) {
    
    RCLCPP_WARN(this->get_logger(), "Initialize node...");
    
    // QoS init
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

    // Parameters    
    this->declare_parameter("evaluation/lane_id", "");
    if (!this->get_parameter("evaluation/lane_id", param_.lane_id)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get lane_id");
    }
    else {
        RCLCPP_INFO(this->get_logger(), "lane_id: %s", param_.lane_id.c_str());
    }
    this->declare_parameter("evaluation/time_limit", 0.0);
    if (!this->get_parameter("evaluation/time_limit", param_.eval_time_limit)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get time_limit");
    }
    else {
        RCLCPP_INFO(this->get_logger(), "time_limit: %f", param_.eval_time_limit);
    }
    this->declare_parameter("evaluation/lane_departure", 0.0);
    if (!this->get_parameter("evaluation/lane_departure", param_.eval_lane_departure)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get lane_departure");
    }
    else {
        RCLCPP_INFO(this->get_logger(), "lane_departure: %f", param_.eval_lane_departure);
    }
    
    std::string dir(getenv("PWD"));
    std::string csv_path("/resources/csv/evaluation_lane");
    param_.ref_csv_path = dir + csv_path;
    RCLCPP_INFO(this->get_logger(), "ref_csv_path: %s", param_.ref_csv_path.c_str());

    // Subscribers
    s_ego_vehicle_state_ = this->create_subscription<ad_msgs::msg::VehicleOutput>(
        "/ego/vehicle_state", qos_profile, std::bind(&Evaluation::CallbackEgoVehicleState, this, std::placeholders::_1));
    s_limit_speed_ = this->create_subscription<std_msgs::msg::Float32>(
        "/ego/limit_speed", qos_profile, std::bind(&Evaluation::CallbackLimitSpeed, this, std::placeholders::_1));
    s_obstacles_ = this->create_subscription<ad_msgs::msg::Obstacles>(
        "/ego/obstacles", qos_profile, std::bind(&Evaluation::CallbackObstacles, this, std::placeholders::_1));

    // Publishers
    p_text_evaluation_result_ = this->create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>(
        "/text_evaluation_result", qos_profile);

    // Initialize
    Init(this->now());

    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / loop_rate)),
        [this]() { this->Run(this->now()); });        
}

Evaluation::~Evaluation() {}

void Evaluation::Init(const rclcpp::Time& current_time) {   

    // Algorithm
    ptr_evaluation_algorithm_ = std::make_unique<EvaluationAlgorithm>(param_);    
    ptr_evaluation_algorithm_->LoadLaneData();
    ptr_evaluation_algorithm_->SetGoalPoint();

    time_prev_ = current_time.seconds();
    time_start_ = current_time.seconds();
}

void Evaluation::Run(const rclcpp::Time& current_time) {
    double curr_evaluation_time = current_time.seconds();
    time_dt_ = curr_evaluation_time - time_prev_;
    time_prev_ = curr_evaluation_time;    

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get subscribe variables
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    interface::VehicleState ego_vehicle_state; {
        mutex_ego_vehicle_state_.lock();
        ego_vehicle_state = i_ego_vehicle_state_;
        mutex_ego_vehicle_state_.unlock();
    }

    mutex_limit_speed_.lock();
    double limit_speed = i_limit_speed_;
    mutex_limit_speed_.unlock();

    mutex_obstacles_.lock();
    ad_msgs::msg::Obstacles obstacles_msg = i_obstacles_;
    mutex_obstacles_.unlock();
    std::vector<ObstacleState> obstacles;
    for(int i=0; i<obstacles_msg.obstacles.size(); i++) {
        ObstacleState obstacle;
        obstacle.x = obstacles_msg.obstacles[i].x;
        obstacle.y = obstacles_msg.obstacles[i].y;
        obstacle.yaw = obstacles_msg.obstacles[i].yaw;
        obstacles.push_back(obstacle);
    }
    
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Algorithm
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //  

    // Calculate errors
    if (param_is_simulator_on_ == true && param_is_finished_ == false && param_is_limit_speed_on_ == true) {
        ptr_evaluation_algorithm_->CalcErrors(ego_vehicle_state, obstacles,
                                              limit_speed, time_eval_driving_, 
                                              time_dt_);
    }
    else if (param_is_simulator_on_ == false) {
        time_start_ = curr_evaluation_time;
    }

    // Check if the evaluation is finished
    if (param_is_finished_ == false) {
        param_is_finished_ = ptr_evaluation_algorithm_->IsFinished(ego_vehicle_state, 6.0);
        time_eval_driving_ = curr_evaluation_time - time_start_;
    }

    // print time start and curr evaluation time using RCLCPP_WARN_STREAM and change time to string
    std::ostringstream time_start_str;
    time_start_str << time_start_;
    std::ostringstream curr_evaluation_time_str;
    curr_evaluation_time_str << curr_evaluation_time;
        
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update output
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    UpdateEvaluationResult(ego_vehicle_state, limit_speed);

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Publish output
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    Publish(current_time);
}

void Evaluation::UpdateEvaluationResult(const interface::VehicleState& ego_vehicle_state,
                                        const double& limit_speed) {
    o_limit_speed_.data = limit_speed;

    std::string evaluation_info 
        = ptr_evaluation_algorithm_->GetEvaluationInfo(ego_vehicle_state, time_eval_driving_, limit_speed);
    // RCLCPP_INFO_STREAM(this->get_logger(), evaluation_info);
    
    rviz_2d_overlay_msgs::msg::OverlayText evaluationText;
    evaluationText.bg_color.r = 0.0f;
    evaluationText.bg_color.g = 0.0f;
    evaluationText.bg_color.b = 0.0f;
    evaluationText.bg_color.a = 0.0f;

    evaluationText.fg_color.r = 0.9f;
    evaluationText.fg_color.g = 0.9f;
    evaluationText.fg_color.b = 0.9f;
    evaluationText.fg_color.a = 0.7f;
    
    evaluationText.line_width = 1;
    evaluationText.text_size = 9.0;
    evaluationText.text = evaluation_info;

    o_text_evaluation_result_ = evaluationText;
}

void Evaluation::Publish(const rclcpp::Time& current_time) {
    p_text_evaluation_result_->publish(o_text_evaluation_result_);
}

int main(int argc, char **argv) {
    std::string node_name = "evaluation";
    double loop_rate      = 50.0;

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Evaluation>(node_name, loop_rate));
    rclcpp::shutdown();

    return 0;
}
