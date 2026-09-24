cmake_minimum_required(VERSION 3.21)
project(rag_retrieval LANGUAGES CXX)

option(RAG_ENABLE_CUDA "Build the CUDA retrieval backend" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core library
add_library(rag_core src/bank.cpp src/search_cpu.cpp)
target_include_directories(rag_core PUBLIC include)

# Optional CUDA backend
if(RAG_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(rag_core PRIVATE src/search_cuda.cu)
  target_link_libraries(rag_core PUBLIC CUDA::cudart)
  target_compile_definitions(rag_core PUBLIC RAG_HAS_CUDA=1)
  set_target_properties(rag_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

# Command-line application
add_executable(rag_query src/main.cpp)
target_link_libraries(rag_query PRIVATE rag_core)

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
