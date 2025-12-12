# cmake_minimum_required(VERSION 3.10)
# project(ResNetInference)

# set(CMAKE_CXX_STANDARD 17)

# # Set ONNX Runtime path
# set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/ai_infra_week1/onnxruntime-linux-x64-1.18.0")

# # Include directories
# include_directories(${ONNXRUNTIME_ROOT}/include)

# # Link directories
# link_directories(${ONNXRUNTIME_ROOT}/lib)

# # Create executable
# add_executable(inference ./ai_infra_week1/inference.cpp)

# # Link libraries
# target_link_libraries(inference onnxruntime)

# # Set RPATH so the executable can find the shared library
# set_target_properties(inference PROPERTIES
#     BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
#     INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
# )

# # Copy model files to build directory
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ai_infra_week1/simple_resnet.onnx
#           ${CMAKE_CURRENT_SOURCE_DIR}/ai_infra_week1/simple_resnet.onnx.data
#      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})



cmake_minimum_required(VERSION 3.10)
project(operator)
add_subdirectory(operator)
