cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda-matmul LANGUAGES CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

find_package(CUDAToolkit REQUIRED)

add_executable(bench bench.cu)
target_include_directories(bench PRIVATE src)
target_link_libraries(bench PRIVATE CUDA::cublas)

target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v;-lineinfo;-g>)
