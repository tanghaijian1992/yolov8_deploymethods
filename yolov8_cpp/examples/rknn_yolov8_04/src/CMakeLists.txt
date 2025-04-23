cmake_minimum_required(VERSION 3.0.2)
project(yolov8_rknn_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
)

find_package(OpenCV REQUIRED)

catkin_package(CATKIN_DEPENDS 
               ${PACKAGE_DEPENDENCIES} 
               roscpp 
               sensor_msgs 
               cv_bridge)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/librknn_api/include
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rga/include
)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/librknn_api/aarch64
                 ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rga/lib/Linux/aarch64)

#add_executable(yolov8_rknn_ros yolov8_rknn_ros.cpp)
#target_link_libraries(yolov8_rknn_ros 
#                      ${catkin_LIBRARIES} 
#                      ${OpenCV_LIBRARIES} 
#                      ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/librknn_api/aarch64/librknn_api.so 
#                      ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/librknn_api/aarch64/librknnrt.so)

add_executable(rknn_engine_demo rknn_engine_demo.cpp)
target_link_libraries(rknn_engine_demo 
                      ${catkin_LIBRARIES} 
                      ${OpenCV_LIBRARIES} 
                      librknnrt.so
                      librga.so)