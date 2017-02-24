cmake_minimum_required(VERSION 3.5)
project(k_shortest_paths)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    main.cpp)

add_executable(k_shortest_paths ${SOURCE_FILES})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
target_link_libraries(k_shortest_paths mpi)
include_directories(${MPI_INCLUDE_PATH})
