cmake_minimum_required(VERSION 3.5)
project(allscale_matrix LANGUAGES CXX C)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# -- contains dependencies for third party libs
set(ALLSCALE_MATRIX_DEPENDENCIES)

include(defaults)
include(allscale)
include(add_module)

# -- Dependency Settings
include(dependencies/pthread)

find_package(Boost REQUIRED)

find_package(GTest)
if(NOT GTest_FOUND)
  message(STATUS "Adding googletest as third party dependency")
  include(dependencies/googletest)
endif()


find_package(Vc)
if(NOT Vc_FOUND)
  message(STATUS "Adding Vc as third party dependency")
  include(dependencies/vc)
endif()

find_package(Eigen3)
if(NOT Eigen3_FOUND)
  message(STATUS "Adding Eigen3 as third party dependency")
  include(dependencies/eigen3)
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

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${Vc_INCLUDE_DIR})
include_directories(${OpenBLAS_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# -- Compilation
if(BUILD_BENCHMARKS)
  add_module_executable_folder("src/benchmark/" ".cc" "benchmark_" "src/benchmark/include;include" OFF ${ALLSCALE_MATRIX_DEPENDENCIES})
endif()

if(BUILD_EXAMPLES)
  add_module_executable_folder("src/examples/" ".cc" "example_" "include" ON ${ALLSCALE_MATRIX_DEPENDENCIES})
endif()

# -- Testing
add_executable(matrix_test src/test/user/data/matrix.cc)

add_dependencies(matrix_test ${ALLSCALE_MATRIX_DEPENDENCIES})

# -- Test Flags
target_include_directories(matrix_test PUBLIC "include")
target_include_directories(matrix_test PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

# -- Test Libraries
target_link_libraries(matrix_test ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(matrix_test ${GTEST_LIBRARIES})
target_link_libraries(matrix_test ${GTEST_MAIN_LIBRARIES})
target_link_libraries(matrix_test ${Vc_LIBRARIES})
target_link_libraries(matrix_test ${OpenBLAS_LIBRARIES})
