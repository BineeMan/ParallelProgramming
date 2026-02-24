cmake_minimum_required(VERSION 4.0)
project(lab1)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

#add_compile_options(-O2)

add_executable(lab1 lab2.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab1 OpenMP::OpenMP_CXX)
endif()