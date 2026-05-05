cmake_minimum_required(VERSION 3.20)

project(lab5)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

add_executable(lab5 lab5.cpp)

target_link_libraries(lab5
        MPI::MPI_CXX
        Threads::Threads
)

target_compile_options(lab5 PRIVATE -O2)