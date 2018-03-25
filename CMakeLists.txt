cmake_minimum_required (VERSION 3.5)

project(example2_proj)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED) 
find_package(CUDA REQUIRED) 

pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4 
                               gstreamer-sdp-1.0>=1.4 
                               gstreamer-video-1.0>=1.4 
                               gstreamer-app-1.0>=1.4) 

include_directories(${OpenCV_INCLUDE_DIRS}, ${GST_INCLUDE_DIRS}, ${CUDA_INCLUDE_DIRS}, /home/nvidia/labs/lab_6/)
add_definitions(${GSTREAMER_DEFINITIONS})

add_executable(example2_proj helpers.h helpers.cpp example2.cpp)

set(CMAKE_BUILD_TYPE Debug)

MESSAGE( STATUS "GST_INCLUDE_DIRS:         " ${GST_INCLUDE_DIRS} )
MESSAGE( STATUS "GST_LIBRARIES:         " ${GST_LIBRARIES} )

target_link_libraries(example2_proj ${OpenCV_LIBS})
target_link_libraries(example2_proj ${GST_LIBRARIES})
target_link_libraries(example2_proj ${CUDA_LIBRARIES})
