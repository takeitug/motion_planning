cmake_minimum_required(VERSION 3.8)
project(motion_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(FRIClient REQUIRED)
find_package(lbr_fri_idl REQUIRED)
find_package(lbr_fri_ros2 REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ライブラリのビルド
add_library(manipulability_lib
  src/inverse.cpp
  src/manipulability_gradient.cpp
  src/forward.cpp
)
target_include_directories(manipulability_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(manipulability_lib Eigen3)

# 実行ファイルのビルド
add_executable(joint_state_listener src/joint_state_listener.cpp)
target_link_libraries(joint_state_listener manipulability_lib)
ament_target_dependencies(joint_state_listener rclcpp sensor_msgs Eigen3 std_msgs)

add_executable(manipulability_client src/manipulability_client.cpp)
target_link_libraries(manipulability_client manipulability_lib FRIClient::FRIClient)
ament_target_dependencies(manipulability_client 
  rclcpp sensor_msgs Eigen3 std_msgs
  control_msgs
  rclcpp_action
  trajectory_msgs
  controller_manager_msgs
  lbr_fri_idl
  lbr_fri_ros2
)

add_executable(manipulability_client_twist src/manipulability_client_twist.cpp)
target_link_libraries(manipulability_client_twist manipulability_lib FRIClient::FRIClient)
ament_target_dependencies(manipulability_client_twist 
  rclcpp sensor_msgs Eigen3 std_msgs
  control_msgs
  rclcpp_action
  trajectory_msgs
  controller_manager_msgs
  lbr_fri_idl
  lbr_fri_ros2
)

add_executable(manipulability_planner src/manipulability_planner.cpp)
ament_target_dependencies(manipulability_planner rclcpp sensor_msgs Eigen3 std_msgs geometry_msgs)
target_include_directories(manipulability_planner PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(manipulability_planner_sim src/manipulability_planner_sim.cpp)
ament_target_dependencies(manipulability_planner_sim rclcpp sensor_msgs Eigen3 std_msgs geometry_msgs)
target_include_directories(manipulability_planner_sim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ヘッダファイルのインストール
install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# 実行ファイル・ライブラリのインストール
install(TARGETS
  joint_state_listener
  manipulability_client
  manipulability_client_twist
  manipulability_lib
  manipulability_planner
  manipulability_planner_sim
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
