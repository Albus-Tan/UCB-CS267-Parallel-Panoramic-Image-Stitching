add_library(reader STATIC reader.cpp)
target_include_directories(reader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(reader PUBLIC ${OpenCV_LIBS})