# Matrices for AllScale

Provides the "Matrix" data type for the AllScale api.

## Dependencies

* CMake 3.5 (<https://cmake.org/>)
* Allscale API (<https://github.com/allscale/allscale_api>)
* Eigen 3.3 (<http://eigen.tuxfamily.org/index.php?title=Main_Page>)
* Vc 1.3 (<https://github.com/VcDevel/Vc/tree/1.3>)
* A C BLAS implementation (e.g. <https://github.com/xianyi/OpenBLAS/wiki/Installation-Guide>)

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

## The Matrix Type

### api/include/data/matrix.h

This is the main implementation file, it provides the Matrix class,
which is built on top of the Grid container.

## Supported Operations

Let `t` be a type.

### scalar-matrix multiplication

Let A, B be matrices with elements of type `t` and size `m * n`.
Let c be an element of type `t`.

> B <- c * A = A * c
>
> ∀ i < m, j < n
>
> B<sub>i,j</sub> = c * A<sub>i,j</sub>

Where `B` is returned by the scalar-matrix multiplication.

### matrix-matrix multiplication

Let A, B, C be matrices with elements of type `t` and sizes `m * k`, `k * n` and `m * n` respectively.

> C <- A * B
>
> ∀ i < m, j < n
>
> C<sub>i,j</sub> = ∑(x < k)  A<sub>i,x</sub> * B<sub>x,j</sub>

Where `C` is returned by the matrix-matrix multiplication.

### matrix-matrix addition

Let A, B, C be matrices with elements of type `t` and size `m * n`.

> C <- A + B
>
> ∀ i < m, j < n
>
> C<sub>i,j</sub> = A<sub>i,j</sub> + B<sub>i,j</sub>

Where `C` is returned by the matrix-matrix addition.

### matrix-matrix subtraction

Let A, B, C be matrices with elements of type `t` and size `m * n`.

> C <- A - B
>
> ∀ i < m, j < n
>
> C<sub>i,j</sub> = A<sub>i,j</sub> - B<sub>i,j</sub>

Where `C` is returned by the matrix-matrix subtraction.

### matrix negation

Let A, B be matrices with elements of type `t` and size `m * n`.

> B <- -A
>
> ∀ i < m, j < n
>
> B<sub>i,j</sub> = -A<sub>i,j</sub>

Where `B` is returned by the matrix negation.

### transpose

Let A, B be matrices with elements of type `t` and sizes `m * n`, `n * m` respectively.

> B <- A.transpose()
>
> ∀ i < m, j < n
>
> B<sub>j,i</sub> = A<sub>i,j</sub>

Where `B` is returned by the matrix transpose.

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
