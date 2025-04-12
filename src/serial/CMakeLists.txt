cmake_minimum_required(VERSION 3.10)
project(SerialStitching)

# Create an executable for the serial stitching implementation
add_executable(serial_stitching main.cpp)

# Link the OpenCV libraries to the executable
target_link_libraries(serial_stitching PRIVATE ${OpenCV_LIBS} reader)
