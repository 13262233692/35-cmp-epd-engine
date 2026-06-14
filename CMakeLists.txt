cmake_minimum_required(VERSION 3.18)
project(cmp_epd_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_EXAMPLES "Build example applications" ON)
option(USE_TENSORRT "Enable TensorRT inference" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

find_package(CUDA REQUIRED)

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

set(EPD_SOURCES
    src/sg_filter.cpp
    src/tensor_reorganizer.cpp
    src/trt_engine.cpp
    src/epd_engine.cpp
)

set(EPD_HEADERS
    include/cmp_epd/common.h
    include/cmp_epd/sg_filter.h
    include/cmp_epd/tensor_reorganizer.h
    include/cmp_epd/trt_engine.h
    include/cmp_epd/epd_engine.h
)

add_library(cmp_epd SHARED ${EPD_SOURCES} ${EPD_HEADERS})

target_link_libraries(cmp_epd
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    cudart
)

if(BUILD_EXAMPLES)
    add_executable(demo_epd examples/demo_epd.cpp)
    target_link_libraries(demo_epd cmp_epd)
endif()

install(TARGETS cmp_epd
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/cmp_epd DESTINATION include)
