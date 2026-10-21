cmake_minimum_required(VERSION 3.0.2)
project(openshc_adapter)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  dynamic_reconfigure
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  syropod_highlevel_controller
)
find_package(Eigen3 REQUIRED)

catkin_package()
get_filename_component(OPENSHC_ADAPTER_REAL "${CMAKE_CURRENT_SOURCE_DIR}" REALPATH)
get_filename_component(HEXAPOD_PROJECT_ROOT "${OPENSHC_ADAPTER_REAL}/../.." REALPATH)
set(OPENSHC_ROOT ${HEXAPOD_PROJECT_ROOT}/third_party/openshc)
include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${OPENSHC_ROOT}/include)

add_executable(openshc_bridge
  bridge.cpp
  ${OPENSHC_ROOT}/src/model.cpp
  ${OPENSHC_ROOT}/src/walk_controller.cpp
  ${OPENSHC_ROOT}/src/pose_controller.cpp
  ${OPENSHC_ROOT}/src/admittance_controller.cpp
  ${OPENSHC_ROOT}/src/debug_visualiser.cpp
)
target_compile_features(openshc_bridge PRIVATE cxx_std_17)
target_link_libraries(openshc_bridge ${catkin_LIBRARIES})
add_dependencies(openshc_bridge ${catkin_EXPORTED_TARGETS}
  syropod_highlevel_controller_generate_messages_cpp
  syropod_highlevel_controller_gencfg
)
