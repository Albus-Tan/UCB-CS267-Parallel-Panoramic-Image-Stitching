cmake_minimum_required(VERSION 3.10)
project(OpenMPStitching)

# Find OpenMP and append the necessary compiler flags if found
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find the OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create an executable for the OpenMP stitching implementation
add_executable(openmp_stitching main.cpp)

# Link the OpenCV libraries to the executable
target_link_libraries(openmp_stitching ${OpenCV_LIBS})
