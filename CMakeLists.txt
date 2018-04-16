cmake_minimum_required(VERSION 3.5)
project(allscale_matrix LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

include(defaults)
include(allscale)

find_package(Boost REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(Vc REQUIRED)
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

message(STATUS "Eigen: " ${EIGEN3_VERSION})

set(THEADS_PREFER_PTHEAD_FLAG on)
find_package(Threads REQUIRED)

# -- The include folder for benchmarks
set(BENCHMARK_INCLUDES "src/benchmark/include")

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message(WARNING "OpenMP could not be found.")
endif(OPENMP_FOUND)

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
  # -- Get all benchmark files
  file(GLOB_RECURSE BENCH_SOURCES src/benchmark/*.cc)

  # MESSAGE(STATUS "## Adding benchmark targets")
  foreach(BENCH_FILE ${BENCH_SOURCES})
    get_filename_component(BENCH_FILENAME ${BENCH_FILE} NAME)
    string(REPLACE ".cc" "" BENCH_FILENAME ${BENCH_FILENAME})
    set(BENCH_FILENAME bench_${BENCH_FILENAME})

    # MESSAGE(STATUS ${BENCH_FILENAME})

    # -- Add Executable
    add_executable(${BENCH_FILENAME} ${BENCH_FILE})

    # -- Default Includes
    target_include_directories(${BENCH_FILENAME} PUBLIC include)
    target_include_directories(${BENCH_FILENAME} PUBLIC ${BENCHMARK_INCLUDES})

    # -- AllScale Definitions
    if(${BENCH_FILENAME} MATCHES "(.*)allscale(.*)")
      # MESSAGE(STATUS "AllScale Benchmark detected")

      target_compile_definitions(${BENCH_FILENAME} PRIVATE EIGEN_DONT_PARALLELIZE=1)
      target_include_directories(${BENCH_FILENAME} PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

      target_link_libraries(${BENCH_FILENAME} ${CMAKE_THREAD_LIBS_INIT})
      target_link_libraries(${BENCH_FILENAME} ${Vc_LIBRARIES})
      target_link_libraries(${BENCH_FILENAME} ${OpenBLAS_LIBRARIES})

    endif(${BENCH_FILENAME} MATCHES "(.*)allscale(.*)")
  endforeach(BENCH_FILE ${BENCH_SOURCES})
endif()

if(BUILD_EXAMPLES)
  # -- Get all example files
  file(GLOB_RECURSE EXAMPLE_SOURCES src/examples/*.cc)

  # MESSAGE(STATUS "## Adding benchmark targets")
  foreach(EXAMPLE_FILE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_FILENAME ${EXAMPLE_FILE} NAME)
    string(REPLACE ".cc" "" EXAMPLE_FILENAME ${EXAMPLE_FILENAME})
    set(EXAMPLE_FILENAME example_${EXAMPLE_FILENAME})

    MESSAGE(STATUS ${EXAMPLE_FILENAME})

    # -- Add Executable
    add_executable(${EXAMPLE_FILENAME} ${EXAMPLE_FILE})

    # -- Default Includes
    target_include_directories(${EXAMPLE_FILENAME} PUBLIC include)
    target_include_directories(${EXAMPLE_FILENAME} PUBLIC ${EXAMPLEMARK_INCLUDES})

    # -- AllScale Definitions
      # MESSAGE(STATUS "AllScale Benchmark detected")

    target_compile_definitions(${EXAMPLE_FILENAME} PRIVATE EIGEN_DONT_PARALLELIZE=1)
    target_include_directories(${EXAMPLE_FILENAME} PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

    target_link_libraries(${EXAMPLE_FILENAME} ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(${EXAMPLE_FILENAME} ${Vc_LIBRARIES})
    target_link_libraries(${EXAMPLE_FILENAME} ${OpenBLAS_LIBRARIES})

  endforeach(EXAMPLE_FILE ${EXAMPLE_SOURCES})
endif()

# -- Testing
add_executable(matrix_test src/test/user/data/matrix.cc)

# -- Test Flags
target_include_directories(matrix_test PUBLIC include)
target_include_directories(matrix_test PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

# -- Test Libraries
target_link_libraries(matrix_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(matrix_test ${Vc_LIBRARIES})
target_link_libraries(matrix_test ${OpenBLAS_LIBRARIES})
