cmake_minimum_required(VERSION 3.21)
project(rag_retrieval LANGUAGES CXX)

option(RAG_ENABLE_CUDA "Build the CUDA retrieval backend" OFF)
option(RAG_ENABLE_CUDA_TIMING "Collect CUDA event timing" OFF)
option(RAG_ENABLE_NVTX "Enable NVTX profiling annotations" OFF)
option(RAG_ENABLE_TT_METAL "Build the Tenstorrent TT-Metal backend" OFF)
set(RAG_TT_METAL_HOME "" CACHE PATH
    "Pinned TT-Metal source/install root (required with RAG_ENABLE_TT_METAL)")
if((RAG_ENABLE_CUDA_TIMING OR RAG_ENABLE_NVTX) AND NOT RAG_ENABLE_CUDA)
  message(FATAL_ERROR
    "RAG_ENABLE_CUDA_TIMING and RAG_ENABLE_NVTX require RAG_ENABLE_CUDA=ON")
endif()
set(RAG_CUDA_ARCHITECTURES "87" CACHE STRING
    "CUDA architectures for RAG kernels (87 is Jetson AGX Orin)")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core library
add_library(rag_core
  src/bank.cpp
  src/retrieval.cpp
  src/search_cpu.cpp
  src/tt_metal_contract.cpp)
target_include_directories(rag_core PUBLIC include)

# TT-Metal is deliberately opt-in. Its C++ API and build products are tied to
# the exact target-host release, so silently selecting headers from a different
# installation would make benchmark results irreproducible.
if(RAG_ENABLE_TT_METAL)
  if(NOT UNIX)
    message(FATAL_ERROR "RAG_ENABLE_TT_METAL is supported only on the Linux p150a host")
  endif()
  if(NOT RAG_TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME})
      set(RAG_TT_METAL_HOME "$ENV{TT_METAL_HOME}")
    else()
      message(FATAL_ERROR
        "Set RAG_TT_METAL_HOME or TT_METAL_HOME to the pinned TT-Metal release")
    endif()
  endif()
  if(NOT EXISTS "${RAG_TT_METAL_HOME}")
    message(FATAL_ERROR "RAG_TT_METAL_HOME does not exist: ${RAG_TT_METAL_HOME}")
  endif()
  list(PREPEND CMAKE_PREFIX_PATH
    "${RAG_TT_METAL_HOME}/build_Release"
    "${RAG_TT_METAL_HOME}/build_Release/lib/cmake")
  if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED CONFIG)
  endif()
  target_sources(rag_core PRIVATE src/search_tt_metal.cpp)
  target_link_libraries(rag_core PUBLIC TT::Metalium)
  target_compile_definitions(rag_core
    PUBLIC RAG_HAS_TT_METAL=1
    PRIVATE
    RAG_TT_KERNEL_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/kernels")
  add_executable(rag_tt_metal_smoke tt_metal/smoke_main.cpp)
  target_link_libraries(rag_tt_metal_smoke PRIVATE rag_core)
endif()

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
  add_executable(tt_metal_contract_test tests/tt_metal_contract_test.cpp)
  target_link_libraries(tt_metal_contract_test PRIVATE rag_core)
  add_test(NAME tt_metal_contract_test COMMAND tt_metal_contract_test)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME bank_builder COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_builder.py)
    add_test(
      NAME tt_bank_packer
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_SOURCE_DIR}/tests/test_tt_bank_packer.py)
    add_test(
      NAME ttnn_backend_contract
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_SOURCE_DIR}/tests/test_ttnn_backend.py)
    add_test(
      NAME ttnn_bench_contract
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_SOURCE_DIR}/tests/test_ttnn_bench.py)
    add_test(
      NAME tt_metal_reduction_contract
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_SOURCE_DIR}/tests/test_tt_metal_model.py)
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
