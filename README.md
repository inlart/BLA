# Matrices for AllScale

A header-only linear algebra library extension of the AllScale API.

## Dependencies

* CMake 3.5 or later (<https://cmake.org/>)
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
| -DUSE_ASSERT            | ON / OFF        |
| -DBUILD_BENCHMARKS      | ON / OFF        |
| -DBUILD_EXAMPLES        | ON / OFF        |
| -DBUILD_TESTS           | ON / OFF        |
| -DOVERRIDE_ALLSCALE_API | \<path\>        |

CMake will print a warning if benchmarks are built with asserts on or build type debug.
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

* `decomposition.h` - LU (partial and full pivoting) and QR decomposition
* `eigen.h` - eigenvalue / eigenvector solver
* `evaluate.h` - copy matrix expressions to actual matrices
* `expressions.h` - expressions that represent (nested) matrix operations; defined in subfolder `expressions`
* `forward.h` - forward declarations
* `householder.h` - Householder reflection helper
* `iterator_wrapper.h` - iteratorception
* `iterator.h` - MatrixExpression iterator
* `matrix_multiplication.h` - different matrix multiplication approaches
* `operators.h` - C++ operator definitions for MatrixExpressions
* `simplify.h` - MatrixExpression tree simplification
* `traits.h` - type traits
* `transpose.h` - matrix transpose implementations
* `transpositions.h` - a list of permutations
* `types.h` - helper types

## Test for the Matrix Type

This file contains tests for:

* `decomposition.cc` - tests for `decomposition.h`
* `evaluate.cc` - tests for `evaluate.h`
* `expressions.cc` - tests for `expressions.h`
* `matrix_multiplication.cc` - tests for `matrix_multiplication.h`
* `operators.cc` - tests for `operators.h`
* `simplify.cc` - tests for `simplify.h`
* `traits.cc` - tests for `traits.h`
* `transpose.cc` - tests for `transpose.h`

## Benchmarking

* matrix operations
  * add
  * subtraction
  * transpose
  * x
  * rowswap

* matrix multiplication
  * mm
  * mmnt
  * mmtn
  * mmtt
  * submm

* decomposition
  * qrd
  * lud

* simplify
  * simplifyscalarmultiplication
  * simplifysubmatrixmultiplication
  * simplifytranspose

## Examples

* `complex.cc` - complex number matrix multiplication
* `expressions.cc` - nested expressions
* `functions.cc` - functions taking MatrixExpressions
* `gmp.cc` - matrix operations using the GNU Multiple Precision library
* `matrix multiplication.cc` - a simple matrix multiplication
* `pagerank.cc` -  PageRank algorithm implementation

## Include what you use

The include what you use mapping file is located at `iwyu/matrix.imp`.

To use it with CMAKE add the option:
`CXX_INCLUDE_WHAT_YOU_USE="path/to/include-what-you-use;-Xiwyu;--mapping_file=/path/to/project/root/iwyu/matrix.imp"`
