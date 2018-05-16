cmake_minimum_required(VERSION 3.5)
project(allscale_matrix LANGUAGES CXX C)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# -- contains dependencies for third party libs
set(ALLSCALE_MATRIX_DEPENDENCIES)

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

include(defaults)
include(allscale)
include(add_module)

include(CTest)

# -- Dependency Settings
include(dependencies/pthread)

find_package(Boost REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/googletest/googletest)

set(GTEST_LIBRARIES gtest)
set(GTEST_MAIN_LIBRARIES gtest_main)

set(ALLSCALE_MATRIX_DEPENDENCIES gtest ${ALLSCALE_MATRIX_DEPENDENCIES})
set(ALLSCALE_MATRIX_DEPENDENCIES gtest_main ${ALLSCALE_MATRIX_DEPENDENCIES})

#Vc master
if(EXISTS ${PROJECT_SOURCE_DIR}/Vc)
  set(Vc_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/Vc)
  message(STATUS "Vc include dir: ${Vc_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "Vc submodule not found. Did you clone with \"git clone --recursive <url>\"?")
endif()

#Eigen 3.3
if(EXISTS ${PROJECT_SOURCE_DIR}/eigen/Eigen)
  set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/eigen)
else()
  message(FATAL_ERROR "Eigen submodule not found. Did you clone with \"git clone --recursive <url>\"?")
endif()

find_package(OpenBLAS)
if(NOT OpenBLAS_FOUND)
  message(STATUS "Adding OpenBLAS as third party dependency")
  include(dependencies/openblas)
endif()

find_package(GMPXX)

message(STATUS "GMPXX found: ${GMPXX_FOUND}")

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(WARNING "OpenMP could not be found.")
endif()

# -- Check if -march=native is available
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

include_directories(SYSTEM ${GTEST_INCLUDE_DIRS})
include_directories(SYSTEM ${Vc_INCLUDE_DIR})
include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# -- Compilation
if(BUILD_BENCHMARKS)
  add_module_executable_folder("src/benchmark/" ".cc" "benchmark_" "" "src/benchmark/include;include" OFF ${ALLSCALE_MATRIX_DEPENDENCIES})
endif()

if(BUILD_EXAMPLES)
  add_module_executable_folder("src/examples/" ".cc" "example_" "" "include" ON ${ALLSCALE_MATRIX_DEPENDENCIES})
endif()

add_test_folder("src/test/user/data" ".cc" "" "_test" "include" ON ${ALLSCALE_MATRIX_DEPENDENCIES})
