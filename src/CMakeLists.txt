cmake_minimum_required(VERSION 3.10)

project(Parallel-Bandits LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

#include_directories(include)

file(GLOB SOURCES "*.cpp")

add_executable(Parallel-Bandits ${SOURCES})

target_link_libraries(Parallel-Bandits PRIVATE OpenMP::OpenMP_CXX)
