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
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# ライブラリのビルド
add_library(manipulability_lib
  src/inverse.cpp
  src/manipulability_gradient.cpp
)
target_include_directories(manipulability_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(manipulability_lib Eigen3)

# 実行ファイルのビルド
add_executable(joint_state_listener src/joint_state_listener.cpp)
target_link_libraries(joint_state_listener manipulability_lib)
ament_target_dependencies(joint_state_listener rclcpp sensor_msgs Eigen3)

# ヘッダファイルのインストール
install(DIRECTORY include/
  DESTINATION include/
)

# 実行ファイル・ライブラリのインストール
install(TARGETS
  joint_state_listener
  manipulability_inverselib
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
