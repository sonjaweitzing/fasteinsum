cmake_minimum_required(VERSION 3.10)
project(einsum-bmm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -fopenmp")

enable_testing()

add_subdirectory(catch)
add_subdirectory(bmm)

