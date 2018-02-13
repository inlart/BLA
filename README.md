# Matrices for AllScale

Provides the "Matrix" data type for the allscale_api. It consists of:

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
