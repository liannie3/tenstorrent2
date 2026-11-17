cmake_minimum_required(VERSION 3.21)
project(rag_retrieval LANGUAGES CXX)

option(RAG_ENABLE_CUDA "Build the CUDA retrieval backend" OFF)
option(RAG_ENABLE_CUDA_TIMING "Collect CUDA event timing" OFF)
option(RAG_ENABLE_NVTX "Enable NVTX profiling annotations" OFF)
option(RAG_ENABLE_TRACY "Enable Tracy CPU and CUDA/CUPTI profiling" OFF)
if((RAG_ENABLE_CUDA_TIMING OR RAG_ENABLE_NVTX OR RAG_ENABLE_TRACY)
   AND NOT RAG_ENABLE_CUDA)
  message(FATAL_ERROR
    "CUDA timing, NVTX, and Tracy require RAG_ENABLE_CUDA=ON")
endif()
set(RAG_CUDA_ARCHITECTURES "87" CACHE STRING
    "CUDA architectures for RAG kernels (87 is Jetson AGX Orin)")
set(RAG_TRACY_SOURCE_DIR "" CACHE PATH
    "Optional Tracy v0.14.0 source tree used when RAG_ENABLE_TRACY=ON")
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
  if(RAG_ENABLE_CUDA_TIMING)
    target_compile_definitions(rag_core PUBLIC RAG_HAS_CUDA_TIMING=1)
  endif()
  if(RAG_ENABLE_NVTX)
    target_compile_definitions(rag_core PUBLIC RAG_HAS_NVTX=1)
  endif()
  if(RAG_ENABLE_TRACY)
    if(RAG_TRACY_SOURCE_DIR)
      add_subdirectory(
        "${RAG_TRACY_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/tracy"
        EXCLUDE_FROM_ALL)
    else()
      find_package(Tracy 0.14 CONFIG REQUIRED)
    endif()
    if(NOT TARGET Tracy::TracyClient)
      message(FATAL_ERROR "The Tracy package does not provide Tracy::TracyClient")
    endif()
    if(NOT TARGET CUDA::cupti)
      message(FATAL_ERROR "RAG_ENABLE_TRACY requires the CUDA CUPTI development library")
    endif()
    target_link_libraries(rag_core PUBLIC Tracy::TracyClient CUDA::cupti)
    target_compile_definitions(rag_core PUBLIC RAG_HAS_TRACY=1 TRACY_ENABLE=1)
  endif()
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
    add_test(
      NAME telemetry_tools
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_SOURCE_DIR}/tests/test_telemetry.py)
    add_test(
      NAME benchmark_tools
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_SOURCE_DIR}/tests/test_benchmark_tools.py
              $<TARGET_FILE:rag_bench>)
  endif()
endif()
