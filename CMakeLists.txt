cmake_minimum_required(VERSION 2.8)
project( TensorRTDemo )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage test.cpp main.cpp)
target_link_libraries( DisplayImage ${OpenCV_LIBS} )