# Matrices for AllScale

Provides the `Matrix` data type for the AllScale API.

## Dependencies

* CMake 3.5 (<https://cmake.org/>)
* Allscale API (<https://github.com/allscale/allscale_api>)
* OpenBLAS 0.3.0 (<https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide>)

The following dependencies are already included as git submodules:

* Eigen 3.3.4 (<http://eigen.tuxfamily.org/index.php?title=Main_Page>)
* Vc (<https://github.com/VcDevel/Vc>)
* googletest 1.8 (<https://github.com/google/googletest/tree/release-1.8.0>)

## CMake Options

| Option                  | Values          |
| ----------------------- | --------------- |
| -DCMAKE_BUILD_TYPE      | Release / Debug |
| -DBUILD_BENCHMARKS      | ON / OFF        |
| -DBUILD_EXAMPLES        | ON / OFF        |
| -DOVERRIDE_ALLSCALE_API | \<path\>        |

If supported, the flag `-march=native` is set.
CMake creates executables for all files in `src/benchmark` with a `.cc` file extension.
Filenames that contain `allscale` may use the AllScale API.
To parallelize Eigen algorithms the compiler has to support OpenMP.

## Preprocessor directives

Support for the following preprocessor directives:

* `ALLSCALE_NO_FAST_MATH` - if defined, associativity for `double` and `float` will **not** be assumed.

## Header Files

### api/include/data/matrix.h

This is the main header file, it provides the Matrix class,
which is built on top of the AllScale Grid container.
It includes the following header files which are contained in the subfolder `impl`:

* `expressions.h` - expressions that represent matrix operations
* `operators.h` - operator definitions for MatrixExpressions
* `forward.h` - forward declaration for expressions and traits
* `matrix_multiplication.h` - contains different variations of matrix-matrix multiplications
* `traits.h` - type traits

## Test for the Matrix Type

### src/test/user/data/matrix.cc

This file contains tests for:

* `Matrix` - General matrix tests
* `Utility` - Matrix helper functions tests
* `Expression` - MatrixExpression tests
* `Operation` - Matrix operations tests
* `Simplify` - MatrixExpression simplification tests

## Benchmarking

### src/benchmark/mm/

Contains matrix multiplication benchmarks

### src/benchmark/add/

Contains matrix addition benchmarks

### src/benchmark/x/

Contains multiple operations benchmarks

### src/benchmark/mm_n_t/

Contains matrix multiplication benchmarks where the right matrix is transposed

### src/benchmark/mm_t_n/

Contains matrix multiplication benchmarks where the left matrix is transposed

### src/benchmark/mm_t_t/

Contains matrix multiplication benchmarks where both matrices are transposed

### src/benchmark/simplify_scalarmultiplication/

Contains scalar matrix multiplication simplification benchmarks

### src/benchmark/simplify_transpose/

Contains transpose simplification benchmarks

### src/benchmark/transpose/

Contains transpose benchmarks

## Include what you use

The include what you use mapping file is located at `iwyu/matrix.imp`.

To use it with CMAKE add the option:
`CXX_INCLUDE_WHAT_YOU_USE="path/to/include-what-you-use;-Xiwyu;--mapping_file=/path/to/project/root/iwyu/matrix.imp"`
