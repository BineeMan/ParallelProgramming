cmake_minimum_required(VERSION 4.0)
project(lab2)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

add_compile_options(-O2)

add_executable(lab2 lab2.cpp)

# Более короткая и современная запись
target_link_libraries(lab2 PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(lab2 PRIVATE -O2)