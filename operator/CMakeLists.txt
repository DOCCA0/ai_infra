cmake_minimum_required(VERSION 3.10)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "89")
endif()

project(OperatorTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include current directory
include_directories(.)

# Create single executable from main.cpp
add_executable(test_operator main.cpp)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Treat main.cpp as a CUDA file so nvcc handles the kernels
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

# Link against CUDA runtime
target_link_libraries(test_operator PRIVATE CUDA::cudart)