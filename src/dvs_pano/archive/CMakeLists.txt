cmake_minimum_required(VERSION 3.0.2)
project(dvs_pano
LANGUAGES C CXX CUDA)

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

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES LBFGSppROS
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O0")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O0")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G") # enable cuda-gdb
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo") # enable line info
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_86 -gencode  arch=compute_86,code=sm_86)

# CUDA
find_package(CUDA 12.6 REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
include_directories("${CUDA_INCLUDE_DIRS}")
# CUDA source files
cuda_add_library(cuda_functions src/add_parallel_kernel.cu)

# # jetson-utils
# find_package(jetson-utils REQUIRED)
include_directories(/usr/lib/aarch64-linux-gnu/tegra)
link_directories(/usr/lib/aarch64-linux-gnu/tegra)

add_executable(${PROJECT_NAME} src/talker.cpp src/dvs_pano.cpp src/pinhole_camera.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} cuda_functions)

include(GoogleTest)
find_package(GTest REQUIRED)
enable_testing()

################################
# Unit Tests
#################################

add_executable(
  unit_test
  src/unit_test.cpp
)

target_link_libraries(
  unit_test
  cuda_functions
  GTest::gtest_main
)

add_test(NAME CudaTest COMMAND unit_test)
