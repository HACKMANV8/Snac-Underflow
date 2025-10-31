cmake_minimum_required(VERSION 3.18)
project(vgmm_demo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCES
    demo.cpp
    vgmm_controller.cpp
    scheduler.cpp
    prefetcher.cpp
    simple_kernel.cu
    logger.cpp
)

add_executable(vgmm_demo ${SOURCES})
target_include_directories(vgmm_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(vgmm_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(vgmm_demo PRIVATE CUDA::cublas)



