# Automotive_software_midterm
차량소프트웨어(mid-term)/ROS2
<Simulator Architecture>
![image](https://github.com/user-attachments/assets/381ed562-ffc8-4a3d-a193-92d8c1045215)

**[Missons]**
### 1. Lane Detection
   - Lane points will be provided not as an array
   - Lane points need to be separated into individual lanes
   - Not all the lanes are solid lanes
   - There will be dashed-lanes
   - There will be a little noise on lane points
### 2. Smart Cruise Control
   - There will be some vehicles on the road
   - The vehicles are slow and will block the road
   - We should follow the vehicle in front of us without collision

**[Strategy]**
### 1. Lane Detection
   -> DBSCAN으로 클러스터링 후, noise를 제거하기 위해 RANSAC 알고리즘 사용
   -> 분류된 lane points들 사용해 3차 방정식 lane_curve_fitting(pseudo inverse matrix 사용)
   -> left_lane/right_lane 피팅 후, lane tracking할 mid_lane 추출
### 2. Smart Cruise Control
   -> Reference velocity PID로 조절
   -> 장애물과의 거리를 기준으로 PID 조절

