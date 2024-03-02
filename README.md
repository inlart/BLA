# BLA Linear Algebra

[![CI](https://github.com/inlart/BLA/workflows/CI/badge.svg)](https://github.com/inlart/BLA/actions?query=workflow%3ACI)

A header-only linear algebra library with optimized expression evaluation and parallelism.

## Dependencies

* [meson](https://mesonbuild.com)
* [ninja](https://ninja-build.org)
* [Allscale API (git)](https://github.com/allscale/allscale_api)
* [Vc 1.4](https://github.com/VcDevel/Vc)
* A C BLAS implementation (e.g. [OpenBLAS](https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide))

For benchmarks/testing:

* [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* [googletest](https://github.com/google/googletest)
* [googlebenchmark](https://github.com/google/benchmark)

## Getting Started

This repository includes the AllScale API as a submodule and should therefore be cloned using
`git clone --recurse-submodules https://github.com/inlart/BLA.git`.

All other dependencies are usually available via a package manager.
e.g. on Ubuntu the requires dependencies can be installed using
`sudo apt install libeigen3-dev libopenblas-dev libgtest-dev vc-dev`.
Note: Instead of `libopenblas-dev` any other package that provides a CBLAS implementation can be used (e.g. `libmkl-dev`).
The preferred library has to be set using the `cblas-library` meson build option.

The build dependencies meson and ninja can be installed using the Python package manager `pip install meson ninja`.

To configure the build run `meson setup build`. Running `ninja` inside the created `build` directory will then build the source.

## Build Options

| Option                  | Values          | Default          |
| ----------------------- | --------------- |------------------|
| enable-examples         | true/false      | false            |
| enable-tests            | true/false      | true             |
| enable-benchmarks       | true/false      | false            |
| cblas-library           | CBLAS library   | openblas         |

Default values can be changed with `meson configure`.

## Preprocessor directives

Support for the following preprocessor directives:

* `BLA_NO_FAST_MATH` - if defined, associativity for `double` and `float` will **not** be assumed.
* `BLA_NO_ETO` - if defined, expression tree optimizations will be disabled

## Development

`bla/matrix.h` is the main header file, it provides the Matrix class which is built on top of the AllScale Grid container.
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
