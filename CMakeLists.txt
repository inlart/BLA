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

set(ALLSCALE_MATRIX_DEPENDENCIES "")

#GTest
if(BUILD_TESTS)
  include(dependencies/googletest)

  set(GTEST_LIBRARIES gtest)
  set(GTEST_MAIN_LIBRARIES gtest_main)

  set(ALLSCALE_MATRIX_DEPENDENCIES gtest ${ALLSCALE_MATRIX_DEPENDENCIES})
  set(ALLSCALE_MATRIX_DEPENDENCIES gtest_main ${ALLSCALE_MATRIX_DEPENDENCIES})
endif()

#GBenchmark
find_package(benchmark)


#Vc 1.3
find_package(Vc REQUIRED)

#Eigen 3.3
if(EXISTS ${EIGEN3_INCLUDE_DIR}/Eigen)
  message(STATUS "EIGEN3_INCLUDE_DIR = ${EIGEN3_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "Eigen3 folder not found. Please set \"EIGEN3_INCLUDE_DIR\"=${EIGEN3_INCLUDE_DIR} correctly.")
endif()

#OpenBLAS
find_package(OpenBLAS)
if(NOT OpenBLAS_FOUND)
  message(WARNING "Adding OpenBLAS as third party dependency. This will rebuild OpenBLAS each time you clean the build directory.")
  include(dependencies/openblas)
endif()

find_package(GMPXX)

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
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
include_directories(SYSTEM ${Vc_INCLUDE_DIR})
include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIRS})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# -- Compilation
if(BUILD_BENCHMARKS)
  add_module_executable_folder("src/benchmark/" ".cc" "benchmark_" "" "src/benchmark/include;include" OFF "${ALLSCALE_MATRIX_DEPENDENCIES}")
endif()

if(BUILD_EXAMPLES)
  add_module_executable_folder("src/examples/" ".cc" "example_" "" "include" ON "${ALLSCALE_MATRIX_DEPENDENCIES}")
endif()

if(BUILD_TESTS)
  add_test_folder("src/test/user/data" ".cc" "" "_test" "include" ON "${ALLSCALE_MATRIX_DEPENDENCIES}")
endif()

if(BUILD_BENCHMARKS AND USE_ASSERT)
  message(WARNING "Building benchmark executables with asserts will falsify the measured times")
endif()

string( TOLOWER "${CMAKE_BUILD_TYPE}" release_type_lower )
if(BUILD_BENCHMARKS AND (NOT release_type_lower STREQUAL "release"))
  message(WARNING "Building benchmark executables without cmake build type release will falsify the measured times")
endif()
