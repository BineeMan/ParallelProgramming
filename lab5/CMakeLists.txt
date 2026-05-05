cmake_minimum_required(VERSION 4.0)
project(lab4)

set(CMAKE_CXX_STANDARD 20)
find_package(MPI REQUIRED)

#add_compile_options(-O2)

add_executable(lab5 lab5.cpp)

target_link_libraries(lab5 MPI::MPI_CXX)
