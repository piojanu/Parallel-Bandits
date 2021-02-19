cmake_minimum_required(VERSION 3.10)

project(Parallel-Bandits LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


##### GoogleTest #####
# Source: https://crascit.com/2015/07/25/cmake-gtest/

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
)
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googletest-download"
)

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds the following targets:
# gtest, gtest_main, gmock and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
                 "${CMAKE_BINARY_DIR}/googletest-build"
)

# The gtest/gmock targets carry header search path dependencies
# automatically when using CMake 2.8.11 or later. Otherwise we
# have to add them here ourselves.
if(CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include"
                        "${gmock_SOURCE_DIR}/include"
    )
endif()


##### OpenMP #####

find_package(OpenMP REQUIRED)


##### IceCream #####
# Source: https://github.com/renatoGarcia/icecream-cpp

find_package(IcecreamCpp)
include_directories(${IcecreamCpp_INCLUDE_DIRS})


#####

include_directories(source)
file(GLOB SOURCES "source/*")
file(GLOB TEST_SOURCES "tests/*")

add_executable(run_main main.cpp ${SOURCES})
target_link_libraries(run_main PRIVATE OpenMP::OpenMP_CXX)

add_executable(run_tests ${TEST_SOURCES} ${SOURCES})
target_link_libraries(run_tests PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(run_tests PRIVATE gtest_main)

enable_testing()
add_test(NAME test_all COMMAND run_tests)
