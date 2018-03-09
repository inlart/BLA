# Matrices for AllScale

Provides the "Matrix" data type for the AllScale api.

## Dependencies

* CMake 3.5 (<https://cmake.org/>)
* Allscale API (<https://github.com/allscale/allscale_api>)
* Eigen 3.3 (<http://eigen.tuxfamily.org/index.php?title=Main_Page>)
* Vc 1.3 (<https://github.com/VcDevel/Vc/tree/1.3>)

## CMake Options

| Option                  | Values          |
| ----------------------- | --------------- |
| -DCMAKE_BUILD_TYPE      | Release / Debug |
| -DOVERRIDE_ALLSCALE_API | \<path\>        |

If supported, the flag `-march=native` is set.
CMake creates executables for all files in `src/benchmark` with a `.cc` file extension.
Filenames that contain `allscale` may use the AllScale API.
To parallelize Eigen algorithms the compiler has to support OpenMP.

## The Matrix Type

### api/include/data/matrix.h

This is the main implementation file, it provides the Matrix class,
which is built on top of the Grid container.

## Test for the Matrix Type

### src/test/user/data/matrix.cc

Test for the Matrix data type.

## Benchmarking

### src/benchmark/mm/

Contains matrix multiplication benchmarks

### src/benchmark/add/

Contains matrix addition benchmarks

### src/benchmark/x/

Contains multiple operations benchmarks
