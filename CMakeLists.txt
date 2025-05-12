cmake_minimum_required(VERSION 3.10)
project(mnist_ffnn LANGUAGES CXX CUDA)
add_executable(mnist_ffnn mnist_ffnn.cu)

# Set c++ standard to c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set target architecture
set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)

# Include CUDA directory
# include_directories(/usr/local/cuda/include)

add_library(Read_mnist read_mnist.cpp)
add_library(NN_kernels nn_kernels.cu)  
add_library(Utils utils.cpp) 

target_link_libraries(mnist_ffnn PUBLIC Read_mnist NN_kernels Utils)

target_include_directories(mnist_ffnn PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}"
                          )