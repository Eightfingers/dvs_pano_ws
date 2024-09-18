cmake_minimum_required(VERSION 3.0.2)
project(dvs_pano)

## Find catkin and any catkin packages

find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS 
  roscpp
  rospy
  std_msgs
  dvs_msgs
  sensor_msgs
  image_transport
  image_geometry
  camera_info_manager
  cv_bridge
)

## Declare a catkin package
catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(dvs_pano src/main.cpp src/pinhole_camera.cpp src/dvs_pano.cpp)
target_link_libraries(dvs_pano ${catkin_LIBRARIES})

