cmake_minimum_required(VERSION 3.10)
project(OperatorTest)

set(CMAKE_CXX_STANDARD 17)

# Include current directory
include_directories(.)

# Create single executable from main.cpp
add_executable(test_operator main.cpp)