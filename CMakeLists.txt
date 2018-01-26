cmake_minimum_required(VERSION 3.5)
project(allscale_matrix LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

include(defaults)
include(allscale)

find_package(Boost REQUIRED)
find_package(GTest REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(THEADS_PREFER_PTHEAD_FLAG on)
find_package(Threads REQUIRED)

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message(WARNING "OpenMP could not be found.")
endif(OPENMP_FOUND)

include_directories(${GTEST_INCLUDE_DIRS})

# -- Compilation

add_executable(mm_allscale src/benchmark/mm_allscale.cc)
add_executable(mm_allscale_strassen src/benchmark/mm_allscale_strassen.cc)
add_executable(mm_eigen src/benchmark/mm_eigen.cc)

add_executable(matrix_test src/test/user/data/matrix.cc)

target_compile_definitions(mm_allscale PRIVATE EIGEN_DONT_PARALLELIZE=1)
target_compile_definitions(mm_allscale_strassen PRIVATE EIGEN_DONT_PARALLELIZE=1)

target_include_directories(mm_allscale PUBLIC include)
target_include_directories(mm_allscale PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

target_include_directories(mm_allscale_strassen PUBLIC include)
target_include_directories(mm_allscale_strassen PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

target_include_directories(mm_eigen PUBLIC include)

target_include_directories(matrix_test PUBLIC include)
target_include_directories(matrix_test PUBLIC ${ALLSCALE_API_INCLUDE_PATH})

target_link_libraries (mm_allscale Eigen3::Eigen)
target_link_libraries (mm_allscale ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries (mm_allscale_strassen Eigen3::Eigen)
target_link_libraries (mm_allscale_strassen ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries (mm_eigen Eigen3::Eigen)

target_link_libraries (matrix_test Eigen3::Eigen)

target_link_libraries(matrix_test ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
