cmake_minimum_required(VERSION 2.8.3)
project(uav_localize)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  dynamic_reconfigure
  message_generation
  mrs_msgs
  tf2
  tf2_msgs
  tf2_sensor_msgs
  tf2_eigen
  tf2_geometry_msgs
  cv_bridge
  image_geometry
  mrs_lib
  nodelet
  uav_detect
  uav_track
)

set(CMAKE_CXX_FLAGS "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-inline-functions")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-inline-functions")
# set(CMAKE_CXX_FLAGS_RELEASE "-g -O0 -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fno-inline-functions")
set(CMAKE_CXX_FLAGS_RELEASE "-Werror -O3")

add_compile_options("-std=c++17" "-Wall" "-Wextra")

find_package(OpenCV REQUIRED)

# include Eigen3
find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

add_message_files(DIRECTORY msgs FILES
  LocalizationHypothesis.msg
  LocalizationHypotheses.msg
)

generate_messages(DEPENDENCIES
  std_msgs
  sensor_msgs
)

generate_dynamic_reconfigure_options(
  config/LocalizationParams.cfg
  )

catkin_package(
  CATKIN_DEPENDS message_runtime mrs_lib # roscpp rospy std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  # ~/git/darknet_ocl/src
)

add_library(LocalizeSingle src/localize_single_nodelet.cpp src/Hypothesis.cpp)
add_dependencies(LocalizeSingle
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(LocalizeSingle
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(backproject_display_node src/backproject_display_node.cpp)
add_dependencies(backproject_display_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)
target_link_libraries(backproject_display_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)
