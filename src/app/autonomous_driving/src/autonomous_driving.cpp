/**
 * @copyright Hanyang University, Department of Automotive Engineering, 2024. All rights reserved.
 *            Subject to limited distribution and restricted disclosure only.
 *
 * @file      autonomous_driving.cpp
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

#include "autonomous_driving.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "limits"
#include <vector>
#include <algorithm>
#include <fstream>

AutonomousDriving::AutonomousDriving(const std::string &node_name, const double &loop_rate,
                                     const rclcpp::NodeOptions &options)
    : Node(node_name, options) {

    RCLCPP_WARN(this->get_logger(), "Initialize node...");

    // QoS init
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

    // Parameters
    this->declare_parameter("autonomous_driving/ns", "");
    if (!this->get_parameter("autonomous_driving/ns", param_vehicle_namespace_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get vehicle_namespace");
        param_vehicle_namespace_ = "";
    }
    else {
        RCLCPP_INFO(this->get_logger(), "param_vehicle_namespace_: %s", param_vehicle_namespace_.c_str());
    }
    this->declare_parameter("autonomous_driving/use_manual_inputs", false);
    if (!this->get_parameter("autonomous_driving/use_manual_inputs", param_use_manual_inputs_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get use_manual_inputs");
        param_use_manual_inputs_ = true;
    }
    else {
        RCLCPP_INFO(this->get_logger(), "param_use_manual_inputs_: %d", param_use_manual_inputs_);
    }
    this->declare_parameter("autonomous_driving/pure_pursuit_kd", 1.0);
    if (!this->get_parameter("autonomous_driving/pure_pursuit_kd", param_pp_kd_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get pure_pursuit_kd");
        param_pp_kd_ = 1.0;
    }
    else {
        RCLCPP_INFO(this->get_logger(), "param_pp_kd_: %f", param_pp_kd_);
    }
    this->declare_parameter("autonomous_driving/pure_pursuit_kv", 0.0);
    if (!this->get_parameter("autonomous_driving/pure_pursuit_kv", param_pp_kv_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get pure_pursuit_kv");
        param_pp_kv_ = 0.0;
    }
    this->declare_parameter("autonomous_driving/pure_pursuit_kc", 0.0);
    if (!this->get_parameter("autonomous_driving/pure_pursuit_kc", param_pp_kc_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get pure_pursuit_kc");
        param_pp_kc_ = 0.0;
    }
    else {
        RCLCPP_INFO(this->get_logger(), "param_pp_kv_: %f", param_pp_kv_);
    }
    this->declare_parameter("autonomous_driving/pid_kp", 0.0);
    if (!this->get_parameter("autonomous_driving/pid_kp", param_pid_kp_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get pid_kp");
        param_pid_kp_ = 0.0;
    }
    else {
        RCLCPP_INFO(this->get_logger(), "param_pid_kp_: %f", param_pid_kp_);
    }
    this->declare_parameter("autonomous_driving/pid_ki", 0.0);
    if (!this->get_parameter("autonomous_driving/pid_ki", param_pid_ki_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get pid_ki");
        param_pid_ki_ = 0.0;
    }
    else {
        RCLCPP_INFO(this->get_logger(), "param_pid_ki_: %f", param_pid_ki_);
    }
    this->declare_parameter("autonomous_driving/pid_kd", 0.0);
    if (!this->get_parameter("autonomous_driving/pid_kd", param_pid_kd_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get pid_kd");
        param_pid_kd_ = 0.0;
    }
    else {
        RCLCPP_INFO(this->get_logger(), "param_pid_kd_: %f", param_pid_kd_);
    }
    this->declare_parameter("autonomous_driving/brake_ratio", 1.0);
    if (!this->get_parameter("autonomous_driving/brake_ratio", param_brake_ratio_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get brake_ratio");
        param_brake_ratio_ = 0.0;
    }
    else {
        RCLCPP_INFO(this->get_logger(), "param_brake_ratio_: %f", param_brake_ratio_);
    }

    // Subscribers
    s_manual_input_ = this->create_subscription<ad_msgs::msg::VehicleInput>(
        "/manual_input", qos_profile, std::bind(&AutonomousDriving::CallbackManualInput, this, std::placeholders::_1));
    s_vehicle_state_ = this->create_subscription<ad_msgs::msg::VehicleOutput>(
        "/ego/vehicle_state", qos_profile, std::bind(&AutonomousDriving::CallbackVehicleState, this, std::placeholders::_1));
    s_limit_speed_ = this->create_subscription<std_msgs::msg::Float32>(
        "/ego/limit_speed", qos_profile, std::bind(&AutonomousDriving::CallbackLimitSpeed, this, std::placeholders::_1));
    s_lane_points_ = this->create_subscription<ad_msgs::msg::LanePointData>(
        "/ego/lane_points", qos_profile, std::bind(&AutonomousDriving::CallbackLanePoints, this, std::placeholders::_1));
    s_obstacles_ = this->create_subscription<ad_msgs::msg::Obstacles>(
        "/ego/obstacles", qos_profile, std::bind(&AutonomousDriving::CallbackObstacles, this, std::placeholders::_1));

    // Publishers
    p_vehicle_command_ = this->create_publisher<ad_msgs::msg::VehicleInput>(
        "/ego/vehicle_command", qos_profile);
    p_driving_way_ = this->create_publisher<ad_msgs::msg::PolyfitLaneData>(
        "/ego/driving_way", qos_profile);
    p_poly_lanes_ = this->create_publisher<ad_msgs::msg::PolyfitLaneDataArray>(
        "/ego/poly_lanes", qos_profile);

    p_cluster_visualization_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/ego/cluster_visualization", qos_profile);

    // Initialize
    Init(this->now());

    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / loop_rate)),
        [this]()
        { this->Run(this->now()); });


    // My code
    lane_point_LEFT.id = "1";
    lane_point_RIGHT.id = "2";
    inliers_LEFT.id = "1";
    inliers_RIGHT.id = "2";
}

AutonomousDriving::~AutonomousDriving() {}

void AutonomousDriving::Init(const rclcpp::Time &current_time) {
}

void AutonomousDriving::UpdateParameter() {
}

double AutonomousDriving::computePID(double target_speed, double current_speed, const rclcpp::Time& current_time, double kp)
{
    double speed_error = target_speed - current_speed;
    double dt = (current_time.seconds() - last_time_.seconds());
    if (dt <= 0.0) {
        dt = 1e-6;
    }

    // PID 계산
    double p_term = kp * speed_error;
    speed_error_integral_ += param_pid_ki_ * speed_error * dt;
    speed_error_integral_ = std::clamp(speed_error_integral_, -max_brake, max_accel);  // 적분 항 한계 설정

    double d_term = param_pid_kd_ * (speed_error - speed_error_prev_) / dt;
    double control_signal = p_term + speed_error_integral_ + d_term;

    // 과거 값 저장
    speed_error_prev_ = speed_error;
    last_time_ = current_time;

    return control_signal;
}


double AutonomousDriving::calculateCurvature(double x_look_ahead) {
    double a = A_MID(3);
    double b = A_MID(2);
    double c = A_MID(1);

    double numerator = std::abs(2 * b + 6 * c * x_look_ahead + 12 * a * std::pow(x_look_ahead, 2));
    double denominator = std::pow(1 + std::pow(3 * a * std::pow(x_look_ahead, 2) + 2 * b * x_look_ahead + c, 2), 1.5);
    return numerator / denominator;
}


double AutonomousDriving::computePID_ACC(double target_distance, double obstacle_distance, const rclcpp::Time& current_time, double kp)
{
    double distance_error = obstacle_distance - target_distance;
    double dt = (current_time.seconds() - last_time_.seconds());
    if (dt <= 0.0) {
        dt = 1e-6;
    }

    // PID 계산
    double p_term = kp * distance_error;
    distance_error_integral_ += param_pid_ki_ * distance_error * dt;
    distance_error_integral_ = std::clamp(distance_error_integral_, -max_brake, max_accel);  // 적분 항 한계 설정

    double d_term = param_pid_kd_ * (distance_error - distance_error_prev_) / dt;
    double control_signal = p_term + distance_error_integral_ + d_term;

    // 과거 값 저장
    distance_error_prev_ = distance_error;
    last_time_ = current_time;

    return control_signal;
}


visualization_msgs::msg::MarkerArray AutonomousDriving::createClusterMarkers(
    const std::vector<Point>& points, int cluster_id) {
    
    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::Marker marker;

    marker.header.frame_id = "ego/body";
    marker.ns = "cluster";
    marker.id = cluster_id;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 1;
    marker.scale.y = 1;
    marker.scale.z = 1;

    // 클러스터마다 다른 색상 적용
    std_msgs::msg::ColorRGBA color;
    color.a = 1.0;
    color.r = static_cast<float>(cluster_id % 3) / 3.0;
    color.g = static_cast<float>((cluster_id + 1) % 3) / 3.0;
    color.b = static_cast<float>((cluster_id + 2) % 3) / 3.0;
    marker.color = color;

    // 각 포인트를 마커에 추가
    for (const auto& point : points) {
        if (point.cluster == cluster_id) {
            geometry_msgs::msg::Point p;
            p.x = point.x;
            p.y = point.y;
            p.z = point.z;
            marker.points.push_back(p);
        }
    }

    marker_array.markers.push_back(marker);
    return marker_array;
}


float distance(const Point& a, const Point& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

// regionQuery 함수: 특정 점을 기준으로 epsilon 거리 내의 이웃 포인트들을 찾음
std::vector<int> AutonomousDriving::regionQuery(const std::vector<Point>& points, int pointIdx, float epsilon) {
    std::vector<int> neighbors;
    for (size_t i = 0; i < points.size(); ++i) {
        if (distance(points[pointIdx], points[i]) <= epsilon) {
            neighbors.push_back(i);
        }
    }
    return neighbors;
}

// expandCluster 함수: DBSCAN에서 클러스터 확장 수행
void AutonomousDriving::expandCluster(std::vector<Point>& points, int pointIdx, int clusterID, float epsilon, size_t min_samples) {
    std::vector<int> seeds = regionQuery(points, pointIdx, epsilon);
    if (seeds.size() < min_samples) {
        points[pointIdx].cluster = -1; // 노이즈로 처리
        return;
    }

    // 초기 포인트 설정
    points[pointIdx].cluster = clusterID;

    // 클러스터 확장
    for (size_t i = 0; i < seeds.size(); ++i) {
        int idx = seeds[i];
        if (!points[idx].visited) {
            points[idx].visited = true;
            std::vector<int> result = regionQuery(points, idx, epsilon);
            if (result.size() >= min_samples) {
                seeds.insert(seeds.end(), result.begin(), result.end());
            }
        }
        if (points[idx].cluster == -1) {
            points[idx].cluster = clusterID;
        }
    }
}

// splitLanePoints 함수: DBSCAN으로 좌/우 차선 구분
void AutonomousDriving::splitLanePoints(const ad_msgs::msg::LanePointData& lane_points) {
    float epsilon = 3.5;  // 밀도 거리
    size_t min_samples = 3;  // 최소 포인트 개수

    // 1. lane_points를 vector<Point>로 변환
    std::vector<Point> points;
    for (const auto& point : lane_points.point) {
        points.push_back({static_cast<float>(point.x), static_cast<float>(point.y), static_cast<float>(point.z)});
    }

    // 2. y > 0과 y < 0에 따라 가장 가까운 포인트(p0, p1)를 선택
    Point p0, p1;
    bool p0_found = false, p1_found = false;
    float min_distance_p0 = std::numeric_limits<float>::max();
    float min_distance_p1 = std::numeric_limits<float>::max();

    for (const auto& point : points) {
        float distance = std::hypot(point.x, point.y);
        if (point.y > 0 && distance < min_distance_p0) {
            min_distance_p0 = distance;
            p0 = point;
            p0_found = true;
        } else if (point.y < 0 && distance < min_distance_p1) {
            min_distance_p1 = distance;
            p1 = point;
            p1_found = true;
        }
    }

    // 3. 초기 좌/우 차선에 대해 클러스터링을 수행할 준비
    lane_point_LEFT.point.clear();
    lane_point_RIGHT.point.clear();

    // 4. 좌측 차선 DBSCAN 시작 (p0 기준으로 expandCluster 호출)
    if (p0_found) {
        size_t p0_index = std::distance(points.begin(), std::find(points.begin(), points.end(), p0));
        expandCluster(points, p0_index, 0, epsilon, min_samples); // 좌측 클러스터 ID는 0
        geometry_msgs::msg::Point new_point;

        for (const auto& point : points) {
            if (point.cluster == 0) {  // 좌측 차선에 속하는 포인트
                new_point.x = point.x;
                new_point.y = point.y;
                new_point.z = point.z;
                lane_point_LEFT.point.push_back(new_point);
            }
        }
    }

    // 5. 우측 차선 DBSCAN 시작 (p1 기준으로 expandCluster 호출)
    if (p1_found) {
        size_t p1_index = std::distance(points.begin(), std::find(points.begin(), points.end(), p1));
        expandCluster(points, p1_index, 1, epsilon, min_samples); // 우측 클러스터 ID는 1
        geometry_msgs::msg::Point new_point;

        for (const auto& point : points) {
            if (point.cluster == 1) {  // 우측 차선에 속하는 포인트
                
                new_point.x = point.x;
                new_point.y = point.y;
                new_point.z = point.z;
                lane_point_RIGHT.point.push_back(new_point);
            }
        }
    }

    RCLCPP_INFO(this->get_logger(), "Lane points split using DBSCAN with initial points p0 and p1.");
}


std::pair<double, double> AutonomousDriving::computeLineModel(const geometry_msgs::msg::Point& p1, const geometry_msgs::msg::Point& p2) {
    double m = (p2.y - p1.y) / (p2.x - p1.x);  // 기울기 m = (y2 - y1) / (x2 - x1)
    double b = p1.y - m * p1.x;                // y 절편 b = y1 - m * x1
    return {m, b};
}


double AutonomousDriving::pointToLineDistance(const geometry_msgs::msg::Point& point, double m, double b) {
    return std::abs(m * point.x - point.y + b) / std::sqrt(m * m + 1);
}

// // RANSAC 알고리즘을 적용해 직선에 맞는 inlier 포인트만 추출
// void AutonomousDriving::applyRANSAC(ad_msgs::msg::LanePointData& lane_points, ad_msgs::msg::LanePointData& inliers, int maxIterations, double distanceThreshold) {
//     // 포인트 개수 확인
//     // RCLCPP_INFO(this->get_logger(), "[1-2] RANSAC START");

//     if (lane_points.point.size() < 2) {
//         RCLCPP_WARN(this->get_logger(), "Not enough points for RANSAC.");
//         return;
//     }

//     std::vector<geometry_msgs::msg::Point>& points = lane_points.point;
//     std::vector<geometry_msgs::msg::Point> bestInliers;
//     srand(static_cast<unsigned>(time(0)));  // 시드 초기화

//     for (int i = 0; i < maxIterations; ++i) {
//         // 포인트 중에서 랜덤하게 2개 선택
//         int idx1 = rand() % points.size();
//         int idx2 = rand() % points.size();
//         while (idx1 == idx2) idx2 = rand() % points.size(); // 같은 포인트 방지

//         // 직선 모델을 계산
//         auto [m, b] = computeLineModel(points[idx1], points[idx2]);

//         // 인라이어를 찾음
//         std::vector<geometry_msgs::msg::Point> currentInliers;
//         for (const auto& point : points) {
//             if (pointToLineDistance(point, m, b) < distanceThreshold) {
//                 currentInliers.push_back(point);
//             }
//         }

//         // 더 많은 인라이어를 가진 모델을 저장
//         if (currentInliers.size() > bestInliers.size()) {
//             bestInliers = currentInliers;
//         }
//     }

//     // 최종적으로 가장 좋은 inliers를 결과로 반환
//     inliers.point = bestInliers;

//     // 디버깅 로그
//     // RCLCPP_INFO(this->get_logger(), "[1-2] RANSAC: Number of inliers = %zu", inliers.point.size());
// }

void AutonomousDriving::applyRANSAC(ad_msgs::msg::LanePointData& lane_points, ad_msgs::msg::LanePointData& inliers, int maxIterations, double distanceThreshold) {
    if (lane_points.point.size() < 4) {  // 최소 네 개의 포인트가 필요
        RCLCPP_WARN(this->get_logger(), "Not enough points for cubic RANSAC.");
        return;
    }

    std::vector<geometry_msgs::msg::Point>& points = lane_points.point;
    std::vector<geometry_msgs::msg::Point> bestInliers;
    srand(static_cast<unsigned>(time(0)));  // 시드 초기화

    for (int i = 0; i < maxIterations; ++i) {
        // 포인트 중에서 랜덤하게 4개 선택
        int idx1 = rand() % points.size();
        int idx2 = rand() % points.size();
        int idx3 = rand() % points.size();
        int idx4 = rand() % points.size();
        while (idx2 == idx1) idx2 = rand() % points.size();
        while (idx3 == idx1 || idx3 == idx2) idx3 = rand() % points.size();
        while (idx4 == idx1 || idx4 == idx2 || idx4 == idx3) idx4 = rand() % points.size();

        // 3차 함수 모델을 계산
        auto [a, b, c, d] = computeCubicModel(points[idx1], points[idx2], points[idx3], points[idx4]);

        // 인라이어를 찾음
        std::vector<geometry_msgs::msg::Point> currentInliers;
        for (const auto& point : points) {
            if (pointToCubicDistance(point, a, b, c, d) < distanceThreshold) {
                currentInliers.push_back(point);
            }
        }

        // 더 많은 인라이어를 가진 모델을 저장
        if (currentInliers.size() > bestInliers.size()) {
            bestInliers = currentInliers;
        }
    }

    // 최종적으로 가장 좋은 inliers를 결과로 반환
    inliers.point = bestInliers;

    // 디버깅 로그
    RCLCPP_INFO(this->get_logger(), "RANSAC with cubic model: Number of inliers = %zu", inliers.point.size());
}

// 3차 함수 모델을 계산하는 함수
std::tuple<double, double, double, double> AutonomousDriving::computeCubicModel(
    const geometry_msgs::msg::Point& p1,
    const geometry_msgs::msg::Point& p2,
    const geometry_msgs::msg::Point& p3,
    const geometry_msgs::msg::Point& p4) {

    Eigen::Matrix4d X;
    Eigen::Vector4d Y;

    X << std::pow(p1.x, 3), std::pow(p1.x, 2), p1.x, 1,
         std::pow(p2.x, 3), std::pow(p2.x, 2), p2.x, 1,
         std::pow(p3.x, 3), std::pow(p3.x, 2), p3.x, 1,
         std::pow(p4.x, 3), std::pow(p4.x, 2), p4.x, 1;

    Y << p1.y, p2.y, p3.y, p4.y;

    Eigen::Vector4d coeffs = X.colPivHouseholderQr().solve(Y);
    return {coeffs(0), coeffs(1), coeffs(2), coeffs(3)};
}

// 3차 함수 모델과 점 사이의 거리를 계산하는 함수
double AutonomousDriving::pointToCubicDistance(
    const geometry_msgs::msg::Point& point,
    double a, double b, double c, double d) {

    double y_estimated = a * std::pow(point.x, 3) + b * std::pow(point.x, 2) + c * point.x + d;
    return std::abs(point.y - y_estimated);
}

// A를 계산하는 함수
Eigen::Vector4d AutonomousDriving::calculateA(const Eigen::VectorXd& X, const Eigen::VectorXd& Y) {
    // RCLCPP_INFO(this->get_logger(), "[1-2] START Calculating A");

    // X_matrix 구성: X 좌표를 3차 다항식으로 변환
    Eigen::MatrixXd X_matrix(X.size(), 4); // 3차 다항식의 경우 4개의 열
    for (int i = 0; i < X.size(); ++i) {
        X_matrix(i, 0) = std::pow(X(i), 3);  // x^3
        X_matrix(i, 1) = std::pow(X(i), 2);  // x^2
        X_matrix(i, 2) = X(i);               // x
        X_matrix(i, 3) = 1;                  // 상수항
    }

    // 유사역행렬(Pseudo-inverse)을 이용하여 A 계산
    Eigen::Vector4d A = X_matrix.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);

    // RCLCPP_INFO(this->get_logger(), "[1-2] Calculated A");
    return A; // Vector4d로 반환
}


// Eigen::Vector4d를 문자열로 변환하는 헬퍼 함수
std::string AutonomousDriving::Vector4dToString(const Eigen::Vector4d& vec) {
    std::ostringstream oss;
    for (int i = 0; i < 4; ++i) {  // Vector4d는 항상 크기가 4입니다.
        if (i > 0) {
            oss << ", ";
        }
        oss << vec(i);
    }
    return oss.str();
}


// void AutonomousDriving::process_lanes(const ad_msgs::msg::LanePointData& lane_point_LEFT, const ad_msgs::msg::LanePointData& lane_point_RIGHT) {

//     // 좌측 및 우측 점의 개수 확인
//     num_points_LEFT = lane_point_LEFT.point.size();
//     num_points_RIGHT = lane_point_RIGHT.point.size();

//     // 점이 없는 경우 예외 처리
//     if (num_points_LEFT == 0) {
//         RCLCPP_ERROR(this->get_logger(), "Left lane has no points.");
//         return;
//     }

//     if (num_points_RIGHT == 0) {
//         RCLCPP_ERROR(this->get_logger(), "Right lane has no points.");
//         return;
//     }

//     // 전역 변수 크기 초기화
//     X_LEFT.resize(num_points_LEFT);
//     Y_LEFT.resize(num_points_LEFT);
//     X_RIGHT.resize(num_points_RIGHT);
//     Y_RIGHT.resize(num_points_RIGHT);

//     // 좌측 점들 저장
//     for (size_t i = 0; i < num_points_LEFT; ++i) {
//         X_LEFT(i) = lane_point_LEFT.point[i].x;
//         Y_LEFT(i) = lane_point_LEFT.point[i].y;
//     }

//     // 우측 점들 저장
//     for (size_t i = 0; i < num_points_RIGHT; ++i) {
//         X_RIGHT(i) = lane_point_RIGHT.point[i].x;
//         Y_RIGHT(i) = lane_point_RIGHT.point[i].y;
//     }

//     // A_LEFT와 A_RIGHT 계산
//     if (X_LEFT.size() >= 4 && X_RIGHT.size() >= 4) {
//         A_LEFT = calculateA(X_LEFT, Y_LEFT);
//         A_RIGHT = calculateA(X_RIGHT, Y_RIGHT);

//         RCLCPP_INFO(this->get_logger(), "A_LEFT and A_RIGHT calculated.");
//     } else {
//         RCLCPP_ERROR(this->get_logger(), "Not enough points to calculate A_LEFT and A_RIGHT (at least 4 points needed).");
//     }
// }

void AutonomousDriving::process_lanes() {
    // RCLCPP_INFO(this->get_logger(), "[1-3] Processing Lanes");

    // RANSAC을 통해 각 차선의 inliers를 계산합니다.
   
    applyRANSAC(lane_point_LEFT, inliers_LEFT);
    applyRANSAC(lane_point_RIGHT, inliers_RIGHT);


    // 오른쪽 차선의 inliers로부터 A_RIGHT 계산
    if (inliers_RIGHT.point.size() > 0) {
        // x 좌표 기준으로 정렬
        std::sort(inliers_RIGHT.point.begin(), inliers_RIGHT.point.end(),
                  [](const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b) {
                      return a.x < b.x;
                  });

        // 일정 간격으로 포인트를 추출
        size_t interval = inliers_RIGHT.point.size() / number_point;
        num_points_RIGHT = number_point;
        X_RIGHT.resize(num_points_RIGHT);
        Y_RIGHT.resize(num_points_RIGHT);

        for (size_t i = 0; i < number_point; ++i) {
            X_RIGHT(i) = inliers_RIGHT.point[i * interval].x;
            Y_RIGHT(i) = inliers_RIGHT.point[i * interval].y;
        }
        
        A_RIGHT = calculateA(X_RIGHT, Y_RIGHT);
        // RCLCPP_INFO(this->get_logger(), "[1-5] RIGHT LANE A: [%s]", this->Vector4dToString(A_RIGHT).c_str());
    } else {
        RCLCPP_WARN(this->get_logger(), "No inliers found for RIGHT lane.");
    }

    // 왼쪽 차선의 inliers로부터 A_LEFT 계산
    if (inliers_LEFT.point.size() > 0) {
        // x 좌표 기준으로 정렬
        std::sort(inliers_LEFT.point.begin(), inliers_LEFT.point.end(),
                  [](const geometry_msgs::msg::Point& a, const geometry_msgs::msg::Point& b) {
                      return a.x < b.x;
                  });

        // 일정 간격으로 포인트를 추출
        size_t interval = inliers_LEFT.point.size() / number_point;
        num_points_LEFT = number_point;
        X_LEFT.resize(num_points_LEFT);
        Y_LEFT.resize(num_points_LEFT);

        for (size_t i = 0; i < number_point; ++i) {
            X_LEFT(i) = inliers_LEFT.point[i * interval].x;
            Y_LEFT(i) = inliers_LEFT.point[i * interval].y;
        }
        
        A_LEFT = calculateA(X_LEFT, Y_LEFT);
        // RCLCPP_INFO(this->get_logger(), "[1-5] LEFT LANE A: [%s]", this->Vector4dToString(A_LEFT).c_str());
    } else {
        RCLCPP_WARN(this->get_logger(), "No inliers found for LEFT lane.");
    }
}

void AutonomousDriving::populatePolyLanes(ad_msgs::msg::PolyfitLaneDataArray& poly_lanes) {
    // RCLCPP_INFO(this->get_logger(), "[1-3] Poly Lanes Processing START");

    // poly_lanes 초기화
    poly_lanes.polyfitlanes.clear(); // 이전 데이터 초기화

    // 오른쪽 차선 (id = "1", A_RIGHT)
    ad_msgs::msg::PolyfitLaneData lane_right;
    lane_right.id = "1";  // id 설정
    lane_right.a0 = A_RIGHT(3);  // a0 계수
    lane_right.a1 = A_RIGHT(2);  // a1 계수
    lane_right.a2 = A_RIGHT(1);  // a2 계수
    lane_right.a3 = A_RIGHT(0);  // a3 계수
    lane_right.frame_id = "ego/body";  // frame_id 설정
    poly_lanes.polyfitlanes.push_back(lane_right);  // 오른쪽 차선을 polyfitlanes에 추가
    // RCLCPP_INFO(this->get_logger(), "[1-3] RIGHT LANE ADDED: a0 = %f, a1 = %f, a2 = %f, a3 = %f", lane_right.a0, lane_right.a1, lane_right.a2, lane_right.a3);

    // 왼쪽 차선 (id = "2", A_LEFT)
    ad_msgs::msg::PolyfitLaneData lane_left;
    lane_left.id = "2";  // id 설정
    lane_left.a0 = A_LEFT(3);  // a0 계수
    lane_left.a1 = A_LEFT(2);  // a1 계수
    lane_left.a2 = A_LEFT(1);  // a2 계수
    lane_left.a3 = A_LEFT(0);  // a3 계수
    lane_left.frame_id = "ego/body";  // frame_id 설정
    poly_lanes.polyfitlanes.push_back(lane_left);  // 왼쪽 차선을 polyfitlanes에 추가
    // RCLCPP_INFO(this->get_logger(), "[1-3] LEFT LANE ADDED: a0 = %f, a1 = %f, a2 = %f, a3 = %f", lane_left.a0, lane_left.a1, lane_left.a2, lane_left.a3);

    // RCLCPP_INFO(this->get_logger(), "[1-3] Poly Lanes Processing END");
}

void AutonomousDriving::populateCenterLane(ad_msgs::msg::PolyfitLaneData& driving_way) {
    // RCLCPP_INFO(this->get_logger(), "[2-1] Center Lanes Processing START");

    // poly_lanes 초기화
    driving_way.id = "0"; // id 설정 아직 중앙이 0인줄 모름
    A_MID(0) = (A_RIGHT(3) + A_LEFT(3)) / 2.0;  // a0 계수
    A_MID(1) = (A_RIGHT(2) + A_LEFT(2)) / 2.0;  // a1 계수
    A_MID(2) = (A_RIGHT(1) + A_LEFT(1)) / 2.0;  // a2 계수
    A_MID(3) = (A_RIGHT(0) + A_LEFT(0)) / 2.0;  // a3 계수

    driving_way.a0 = A_MID(0);  // a0 계수
    driving_way.a1 = A_MID(1);  // a1 계수
    driving_way.a2 = A_MID(2);  // a2 계수
    driving_way.a3 = A_MID(3);  // a3 계수
    
    // RCLCPP_INFO(this->get_logger(), "[2-1] Driving Way ADDED: a0 = %f, a1 = %f, a2 = %f, a3 = %f", driving_way.a0, driving_way.a1, driving_way.a2, driving_way.a3);

    
    // RCLCPP_INFO(this->get_logger(), "[2-1] Poly Lanes Processing END");
}

double AutonomousDriving::getDistanceToVehicle(const ad_msgs::msg::VehicleOutput& vehicle_state, const ad_msgs::msg::VehicleOutput& obstacle) {
    double dx = obstacle.x - vehicle_state.x;
    double dy = obstacle.y - vehicle_state.y;
    return std::sqrt(dx * dx + dy * dy);
}

void AutonomousDriving::PurePursuit(){
    g_y = A_MID(3) * std::pow(g_x, 3) + A_MID(2) * std::pow(g_x, 2) + A_MID(1) * g_x + A_MID(0);
    l_d = std::sqrt(std::pow(g_x, 2) + std::pow(g_y, 2));
    e_ld = g_y;
    yaw = std::atan2(2 * L * g_y , std::pow(l_d, 2));
}

bool AutonomousDriving::isCurveAhead(double curvature_threshold) {
    // 샘플링할 범위와 간격 설정
    double x_start = 0.0;                // 시작 x 좌표 (예: 현재 위치)
    double x_end = l_d + 5;                  // Look-ahead 거리만큼의 끝 x 좌표
    int num_samples = 10;                // 샘플링할 포인트 개수
    double dx = (x_end - x_start) / num_samples; // 각 샘플링 간격

    double a = A_MID(3); // 3차 항 계수
    double b = A_MID(2); // 2차 항 계수
    double c = A_MID(1); // 1차 항 계수
    double d = A_MID(0); // 0차 항 계수

    // 각 지점에서 곡률을 계산하여 저장
    std::vector<double> curvatures;
    for (int i = 0; i <= num_samples; ++i) {
        double x = x_start + i * dx;

        // 곡률 계산을 위한 분자와 분모
        double numerator = std::abs(6 * a * x + 2 * b);  // y''(x)
        double denominator = std::pow(1 + std::pow(3 * a * std::pow(x, 2) + 2 * b * x + c, 2), 1.5);  // (1 + (y'(x))^2)^(3/2)
        
        double curvature = numerator / denominator;
        curvatures.push_back(curvature);
    }

    // 최대 곡률 계산 (또는 필요에 따라 평균 곡률 사용 가능)
    double max_curvature = *std::max_element(curvatures.begin(), curvatures.end());

    // 곡률이 임계값 이상이면 커브로 판단
    return max_curvature >= curvature_threshold;
}





void AutonomousDriving::Run(const rclcpp::Time &current_time) {
    UpdateParameter();

    visualization_msgs::msg::MarkerArray cluster_markers;
    double AEB_distance = 12;
    double flag = 0.0;
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get subscribe variables
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    mutex_vehicle_state_.lock();
    ad_msgs::msg::VehicleOutput vehicle_state = i_vehicle_state_;
    mutex_vehicle_state_.unlock();

    mutex_obstacles_.lock();
    ad_msgs::msg::Obstacles obstacles = i_obstacles_;
    mutex_obstacles_.unlock();

    mutex_limit_speed_.lock();
    double limit_speed = i_limit_speed_;
    mutex_limit_speed_.unlock();

    mutex_lane_points_.lock();
    ad_msgs::msg::LanePointData lane_points = i_lane_points_;
    mutex_lane_points_.unlock(); 


    double target_speed = 0.0; // m/s
    double target_distance = 0.0;
    double control_signal;

    mutex_obstacles_.lock();
    std::vector<ad_msgs::msg::VehicleOutput> obstacle_list = i_obstacles_.obstacles;
    mutex_obstacles_.unlock();


    if (!obstacle_list.empty()){

        // y 좌표를 기준으로 정렬하여 가운데 장애물 선택
        std::sort(obstacle_list.begin(), obstacle_list.end(), [](const ad_msgs::msg::VehicleOutput& a, const ad_msgs::msg::VehicleOutput& b) {
            return a.y < b.y;
        });

        // 중간에 위치한 장애물 선택
        ad_msgs::msg::VehicleOutput middle_obstacle = obstacle_list[obstacle_list.size() / 2];
        closest_obstacle_distance = getDistanceToVehicle(vehicle_state, middle_obstacle);

    }

    
    // RCLCPP_INFO(this->get_logger(), "limit_speed: %f", /limit_speed);

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Output variables
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    ad_msgs::msg::PolyfitLaneDataArray poly_lanes;
    poly_lanes.frame_id = param_vehicle_namespace_ + "/body";

    ad_msgs::msg::PolyfitLaneData driving_way;
    driving_way.frame_id = param_vehicle_namespace_ + "/body";
    
    ad_msgs::msg::VehicleInput vehicle_command;

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Algorithm
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    /**
     * @brief Calculate the vehicle comamnd using given inputs
     * 
     * inputs
     *      vehicle_state, limit_speed, obstacles, lane_points
     * outputs
     *      driving_way, poly_lanes, vehicle_command
     * 
     */
    if (param_use_manual_inputs_ == false) {
        // RCLCPP_INFO(this->get_logger(), " ************* START ***************");
        // 0. Different to previous practice, the lane point data is not sorted by lanes.
        //    You have to divide the points by their lane.
        lane_point_LEFT.point.clear();
        lane_point_RIGHT.point.clear();
        inliers_LEFT.point.clear();
        inliers_RIGHT.point.clear();
        if (!lane_points.point.empty()){
            splitLanePoints(lane_points);

            // 1. With divieded points, you can curve fit the lane and find the left, right lane.
            //    The generated left and right lane should be stored in the "poly_lanes".
            //    If you do so, the Display node will visualize the lanes.
            number_point = 4;
            // process_lanes(lane_point_LEFT, lane_point_RIGHT);
            process_lanes();
            populatePolyLanes(poly_lanes);
            // 2. Generate the center line(=driving_way) which the vehicle will follow.
            //    The generated center line should be stored in the "driving_way".
            //    If you do so, the Display node will visualize the center line.
            populateCenterLane(driving_way);
            
            // 3. Calculate the longitudinal command(acceleration[0~1] or brake[0~1])
            //    You can tune your controller using the ros parameter.
            //    We provide the example of 'PID' parameters, so you can edit and use them.
            // vehicle_command.accel = 0.0;
            // vehicle_command.brake = 0.0;

            PurePursuit();

            if (!lane_point_LEFT.point.empty() && !lane_point_RIGHT.point.empty()) 
            {
                std::vector<Point> all_points;

                // lane_point_LEFT.point와 lane_point_RIGHT.point의 각 요소를 Point로 변환하여 all_points에 추가
                for (const auto& p : lane_point_LEFT.point) {
                    all_points.emplace_back(p); // 변환 생성자 사용
                }

                for (const auto& p : lane_point_RIGHT.point) {
                    all_points.emplace_back(p); // 변환 생성자 사용
                }

                // 클러스터 ID별로 MarkerArray 생성 후 퍼블리시
                for (int i = 0; i < 2; ++i) {  // 0: LEFT, 1: RIGHT
                    visualization_msgs::msg::MarkerArray cluster_markers = createClusterMarkers(all_points, i);
                    p_cluster_visualization_->publish(cluster_markers);
                }
            }


            bool is_curve_ahead = isCurveAhead();
            if (is_curve_ahead) {
                RCLCPP_INFO(this->get_logger(), "Road state: Curve detected");
            } else {
                RCLCPP_INFO(this->get_logger(), "Road state: Straight road");
            }

            if (is_curve_ahead) {
                target_distance = 10;
                target_speed = 40/3.6;  // 커브일 경우 target_distance를 20으로 설정
                RCLCPP_INFO(this->get_logger(), "Curve detected ahead. Adjusted target distance: %f", target_distance);
            } else {
                target_distance = 10;
                target_speed = 80/3.6;
                RCLCPP_INFO(this->get_logger(), "Straight road ahead. Using default target distance: %f", target_distance);
            }

            if(closest_obstacle_distance > AEB_distance)
            {
                if(closest_obstacle_distance > 15.0){
                    control_signal = computePID(target_speed, vehicle_state.velocity, current_time, 0.6);
                    RCLCPP_INFO(this->get_logger(), "reference speed control");
                } else {
                    control_signal = computePID_ACC(target_distance, closest_obstacle_distance, current_time, 0.25);
                    RCLCPP_INFO(this->get_logger(), "ACC control");
                    flag = 1.0;
                }

                // control_signal = computePID_ACC(target_distance, closest_obstacle_distance, current_time, 0.05);
                
                /* We will provide simple speed control code. You can use your own code */
                if (control_signal > 0)
                {
                    if (flag == 1.0){
                        vehicle_command.accel = std::min(control_signal, 0.3);
                        vehicle_command.brake = 0.0;
                    }
                    vehicle_command.accel = std::min(control_signal, 0.5);
                    vehicle_command.brake = 0.0;
                }
                else
                { 
                    vehicle_command.accel = 0.0;
                    vehicle_command.brake = std::min(-control_signal, max_brake);
                }
            }else{
                RCLCPP_INFO(this->get_logger(), "!!!!AEB!!!!");
                vehicle_command.accel = 0.0;
                vehicle_command.brake = 1.0;
            }
            
            

            
            // 4. Calculate the lateral command (steering angle [rad])
            //    You can tune your controller using the ros parameter.
            //    We provide the example of 'Pure Pursuit' parameters, so you can edit and use them.
            
            vehicle_command.steering = yaw;
            // RCLCPP_INFO(this-> get_logger(), "[3-1] yaw = %f", yaw);
        }
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Update output
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    o_driving_way_ = driving_way;
    o_poly_lanes_ = poly_lanes;
    o_vehicle_command_ = vehicle_command;
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Publish output
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    RCLCPP_INFO(this->get_logger(), "------------------------------------------");
    RCLCPP_INFO(this->get_logger(), "target_distance: %f", target_distance);
    RCLCPP_INFO(this->get_logger(), "lead_distance: %f", closest_obstacle_distance);
    RCLCPP_INFO(this->get_logger(), "Distance_error: %f", target_distance-closest_obstacle_distance);
    Publish(current_time, cluster_markers);
}

void AutonomousDriving::Publish(const rclcpp::Time &current_time, const visualization_msgs::msg::MarkerArray& cluster_markers) {
    p_vehicle_command_->publish(o_vehicle_command_);
    p_driving_way_->publish(o_driving_way_);
    p_poly_lanes_->publish(o_poly_lanes_);
    p_cluster_visualization_->publish(cluster_markers);
}

int main(int argc, char **argv) {
    std::string node_name = "autonomous_driving";
    double loop_rate = 100.0;

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AutonomousDriving>(node_name, loop_rate));
    rclcpp::shutdown();
    return 0;
}
