cmake_minimum_required(VERSION 3.21)
project(rag_retrieval LANGUAGES CXX)

option(RAG_ENABLE_CUDA "Build the CUDA retrieval backend" OFF)
set(RAG_CUDA_ARCHITECTURES "87" CACHE STRING
    "CUDA architectures for RAG kernels (87 is Jetson AGX Orin)")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core library
add_library(rag_core src/bank.cpp src/retrieval.cpp src/search_cpu.cpp)
target_include_directories(rag_core PUBLIC include)

# Optional CUDA backend
if(RAG_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(rag_core PRIVATE src/search_cuda.cu)
  target_link_libraries(rag_core PUBLIC CUDA::cudart CUDA::cublas)
  target_compile_definitions(rag_core PUBLIC RAG_HAS_CUDA=1)
  set_target_properties(rag_core PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${RAG_CUDA_ARCHITECTURES}")
endif()

# Command-line application
add_executable(rag_query src/main.cpp)
target_link_libraries(rag_query PRIVATE rag_core)

# Reusable benchmark runner and command-line harness
add_library(rag_bench_lib src/bench.cpp)
target_link_libraries(rag_bench_lib PUBLIC rag_core)

add_executable(rag_bench bench/bench_main.cpp)
target_link_libraries(rag_bench PRIVATE rag_bench_lib)
if(WIN32)
  target_link_libraries(rag_bench PRIVATE Psapi)
endif()

# Tests
include(CTest)
if(BUILD_TESTING)
  add_executable(rag_test tests/rag_test.cpp)
  target_link_libraries(rag_test PRIVATE rag_core)
  add_test(NAME rag_test COMMAND rag_test)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME bank_builder COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_builder.py)
  endif()
endif()
