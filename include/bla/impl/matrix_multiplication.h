#pragma once

#include <allscale/api/core/prec.h>
#include <allscale/api/user/algorithm/async.h>
#include <allscale/api/user/algorithm/pfor.h>
#include <allscale/api/user/data/grid.h>
#include <allscale/utils/assert.h>
#include <allscale/utils/vector.h>

#include "bla/impl/expressions.h"
#include "bla/impl/operators.h"
#include "bla/impl/simplify.h"
#include "bla/impl/types.h"

#include "bla/impl/forward.h"

// -- Other
#include <Vc/Vc>
#include <array>
#include <cblas.h>

namespace bla {
namespace impl {

namespace detail {

// -- a recursive
template <int Depth = 1024, typename T>
void strassen_rec(SubMatrix<const Matrix<T>> A, SubMatrix<const Matrix<T>> B, SubMatrix<Matrix<T>> C) {
    assert_eq(A.size(), B.size());
    assert_eq(B.size(), C.size());

    static_assert(Depth > 0, "strassen depth has to be > 0");
    // execute base case multiplication if the size is small enough
    if(A.rows() <= Depth) {
        C = A * B;
        return;
    }

    coordinate_type m = A.rows() / 2;
    point_type size_m{m, m};

    SubMatrix<const Matrix<T>> a11 = A.sub({{0, 0}, size_m});
    SubMatrix<const Matrix<T>> a12 = A.sub({{0, m}, size_m});
    SubMatrix<const Matrix<T>> a21 = A.sub({{m, 0}, size_m});
    SubMatrix<const Matrix<T>> a22 = A.sub({size_m, size_m});

    SubMatrix<const Matrix<T>> b11 = B.sub({{0, 0}, size_m});
    SubMatrix<const Matrix<T>> b12 = B.sub({{0, m}, size_m});
    SubMatrix<const Matrix<T>> b21 = B.sub({{m, 0}, size_m});
    SubMatrix<const Matrix<T>> b22 = B.sub({size_m, size_m});

    auto c11 = C.sub({{0, 0}, size_m});
    auto c12 = C.sub({{0, m}, size_m});
    auto c21 = C.sub({{m, 0}, size_m});
    auto c22 = C.sub({size_m, size_m});

    Matrix<T> u1(size_m);
    Matrix<T> u2(size_m);
    Matrix<T> u3(size_m);
    Matrix<T> u4(size_m);
    Matrix<T> u5(size_m);
    Matrix<T> u6(size_m);
    Matrix<T> u7(size_m);

    Matrix<T> s1(size_m);
    Matrix<T> s2(size_m);
    Matrix<T> s3(size_m);
    Matrix<T> s4(size_m);

    Matrix<T> t1(size_m);
    Matrix<T> t2(size_m);
    Matrix<T> t3(size_m);
    Matrix<T> t4(size_m);

    Matrix<T> p1(size_m);
    Matrix<T> p2(size_m);
    Matrix<T> p3(size_m);
    Matrix<T> p4(size_m);
    Matrix<T> p5(size_m);
    Matrix<T> p6(size_m);
    Matrix<T> p7(size_m);

    s1 = a21 + a22;
    s2 = s1 - a11;
    s3 = a11 - a21;
    s4 = a12 - s2;

    t1 = b12 - b11;
    t2 = b22 - t1;
    t3 = b22 - b12;
    t4 = t2 - b21;

    // do the recursive computations
    auto p1_async = allscale::api::user::algorithm::async([&]() { strassen_rec(a11, b11, SubMatrix<Matrix<T>>(p1)); });

    auto p2_async = allscale::api::user::algorithm::async([&]() { strassen_rec(a12, b21, SubMatrix<Matrix<T>>(p2)); });

    auto p3_async = allscale::api::user::algorithm::async([&]() { strassen_rec(SubMatrix<const Matrix<T>>(s4), b22, SubMatrix<Matrix<T>>(p3)); });

    auto p4_async = allscale::api::user::algorithm::async([&]() { strassen_rec(a22, SubMatrix<const Matrix<T>>(t4), SubMatrix<Matrix<T>>(p4)); });

    auto p5_async = allscale::api::user::algorithm::async([&]() { strassen_rec(SubMatrix<const Matrix<T>>(s1), SubMatrix<const Matrix<T>>(t1), SubMatrix<Matrix<T>>(p5)); });

    auto p6_async = allscale::api::user::algorithm::async([&]() { strassen_rec(SubMatrix<const Matrix<T>>(s2), SubMatrix<const Matrix<T>>(t2), SubMatrix<Matrix<T>>(p6)); });

    auto p7_async = allscale::api::user::algorithm::async([&]() { strassen_rec(SubMatrix<const Matrix<T>>(s3), SubMatrix<const Matrix<T>>(t3), SubMatrix<Matrix<T>>(p7)); });

    p1_async.wait();
    p2_async.wait();
    p3_async.wait();
    p4_async.wait();
    p5_async.wait();
    p6_async.wait();
    p7_async.wait();

    u1 = p1 + p2;
    u2 = p1 + p6;
    u3 = u2 + p7;
    u4 = u2 + p5;
    u5 = u4 + p3;
    u6 = u3 - p4;
    u7 = u3 + p5;

    // update the result matrix
    allscale::api::user::algorithm::pfor(size_m, [&](const point_type& p) {
        c11[p] = u1[p];
        c12[p] = u5[p];
        c21[p] = u6[p];
        c22[p] = u7[p];
    });
}

} // namespace detail

#define mindex(i, j, size) ((i) * (size) + (j))

// calculate a size * size block
template <int size = 8, typename T>
void block(point_type end, T* result, const T* lhs, const T* rhs, triple_type matrix_sizes) {
    using ct = coordinate_type;
    using vt = Vc::Vector<T>;

    static_assert(size % vt::size() == 0, "vector type size doesn't divide 'size'"); // our vector type 'vt' fits into the size x size segment

    constexpr int vector_size = size / vt::size(); // vector_size contains the number of vt types needed per line

    const auto k = end.x;

    std::array<const T*, size> lhs_ptr;

    for(ct j = 0; j < size; ++j) {
        lhs_ptr[j] = lhs + mindex(j, 0, matrix_sizes.y);
    }

    std::array<std::array<vt, vector_size>, size> res;

    // initialize the result
    for(int i = 0; i < size; ++i) {
        for(int j = 0; j < vector_size; ++j) {
            res[i][j] = 0;
        }
    }

    for(ct i = 0; i < k; ++i) {
        std::array<vt, size> a;

        // copy the content of the lhs to a
        for(ct j = 0; j < size; ++j) {
            a[j] = *lhs_ptr[j]++;
        }

        std::array<vt, vector_size> b;

        // calculate the result
        for(ct j = 0; j < vector_size; ++j) {
            b[j].load(rhs + j * vt::size() + i * size);


            for(ct jj = 0; jj < size; ++jj) {
                res[jj][j] += a[jj] * b[j];
            }
        }
    }

    // update the result matrix
    for(ct i = 0; i < size; ++i) {
        for(ct j = 0; j < vector_size; ++j) {
            ct jj = j * (ct)vt::size();
            for(ct k = 0; k < (ct)vt::size(); ++k) {
                result[mindex(i, jj + k, matrix_sizes.z)] += res[i][j][k];
            }
        }
    }
}

// -- parallel matrix * matrix multiplication kernel
// template <int size = 8, typename T>
// void kernel(point_type end, T* result, const T* lhs, const T* rhs, triple_type matrix_sizes) {
//     using ct = coordinate_type;

//     T packed_b[end.y * end.x];

//     // copy rhs
//     allscale::api::user::algorithm::pfor(GridPoint<1>{end.y / size}, [&](const auto& pos) {
//         ct j = pos[0] * size;
//         T* b_pos = packed_b + (j * end.x);
//         for(int k = 0; k < end.x; ++k) {
//             for(int jj = 0; jj < size; ++jj) {
//                 *b_pos++ = rhs[mindex(k, jj + j, matrix_sizes.z)];
//             }
//         }
//     });

//     // calculate part that is blockable
//     /*
//      * ######--
//      * ######--
//      * ######--
//      * --------
//      * --------
//      */
//     allscale::api::user::algorithm::pfor(point_type{matrix_sizes.x / size, end.y / size}, [&](const auto& pos) {
//         ct i = pos.x * size;
//         ct j = pos.y * size;

//         block<size>(end, result + mindex(i, j, matrix_sizes.z), lhs + mindex(i, 0, matrix_sizes.y), packed_b + (j * end.x), matrix_sizes);
//     });

//     // calculate part to the right
//     /*
//      * ------##
//      * ------##
//      * ------##
//      * --------
//      * --------
//      */
//     for(ct i = 0; i < matrix_sizes.x - (matrix_sizes.x % size); ++i) {
//         for(ct j = end.y - (end.y % size); j < end.y; ++j) {
//             for(ct k = 0; k < end.x; ++k) {
//                 result[mindex(i, j, matrix_sizes.z)] += lhs[mindex(i, k, matrix_sizes.y)] * rhs[mindex(k, j, matrix_sizes.z)];
//             }
//         }
//     }

//     // calculate bottom part
//     /*
//      * --------
//      * --------
//      * --------
//      * ########
//      * ########
//      */
//     for(ct i = matrix_sizes.x - (matrix_sizes.x % size); i < matrix_sizes.x; ++i) {
//         for(ct j = 0; j < end.y; ++j) {
//             for(ct k = 0; k < end.x; ++k) {
//                 result[mindex(i, j, matrix_sizes.z)] += lhs[mindex(i, k, matrix_sizes.y)] * rhs[mindex(k, j, matrix_sizes.z)];
//             }
//         }
//     }
// }

// -- parallel matrix * matrix multiplication
// template <typename T>
// void matrix_multiplication_allscale(Matrix<T>& result, const Matrix<T>& lhs, const Matrix<T>& rhs) {
//     assert(lhs.columns() == rhs.rows());

//     using ct = coordinate_type;

//     const coordinate_type nc = 512;
//     const coordinate_type kc = 256;

//     const auto m = lhs.rows();
//     const auto k = lhs.columns();
//     const auto n = rhs.columns();

//     constexpr auto size = Vc::Vector<T>::size();

//     // TODO: find good values for kc, nc (multiple of vector size?)

//     result.zero();


//     for(ct kk = 0; kk < k; kk += kc) {
//         ct kb = std::min(k - kk, kc);
//         for(ct j = 0; j < n; j += nc) {
//             ct jb = std::min(n - j, nc);

//             kernel<size>({kb, jb}, &result[{0, j}], &lhs[{0, kk}], &rhs[{kk, j}], {m, k, n});
//         }
//     }
// }

// -- matrix * matrix multiplication using a single BLAS level 3 function call
void matrix_multiplication_blas(Matrix<double>& result, const Matrix<double>& lhs, const Matrix<double>& rhs) {
    assert(lhs.columns() == rhs.rows());

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lhs.rows(), rhs.columns(), lhs.columns(), 1.0, &lhs[{0, 0}], lhs.columns(), &rhs[{0, 0}],
                rhs.columns(), 0.0, &result[{0, 0}], rhs.columns());
}

// -- parallel matrix * matrix multiplication using BLAS level 3 function calls
void matrix_multiplication_pblas(Matrix<double>& result, const Matrix<double>& lhs, const Matrix<double>& rhs) {
    assert(lhs.columns() == rhs.rows());

    auto blas_multiplication = [&](const range_type& r) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, r.y, rhs.columns(), lhs.columns(), 1.0, &lhs[{r.x, 0}], lhs.columns(), &rhs[{0, 0}],
                    rhs.columns(), 0.0, &result[{r.x, 0}], rhs.columns());
    };

    #ifdef BLA_MM_PARALLEL_CUTOFF
        const coordinate_type cutoff = BLA_MM_PARALLEL_CUTOFF;
    #else
        const coordinate_type cutoff = 128;
    #endif

    auto multiplication_rec = allscale::api::core::prec(
        // base case test
        [&](const range_type& r) { return r.y < cutoff; },
        // base case
        blas_multiplication,
        allscale::api::core::pick(
            // parallel recursive split
            [&](const range_type& r, const auto& rec) {
                int mid = r.x + r.y / 2;
                return allscale::api::core::parallel(rec({r.x, r.y / 2}), rec({mid, r.y - r.y / 2}));
            },
            // BLAS multiplication if no further parallelism can be exploited
            [&](const range_type& r, const auto&) {
                blas_multiplication(r);
                return allscale::api::core::done();
            }));

    multiplication_rec({0, lhs.rows()}).wait();
}

// -- parallel block matrix * matrix multiplication using BLAS level 3 function calls
template <bool transLHS = false, bool transRHS = false, typename T, typename Func>
void matrix_multiplication_pbblas(Matrix<T>& result, const Matrix<T>& lhs, const Matrix<T>& rhs, Func f) {
    assert_eq((transLHS ? lhs.rows() : lhs.columns()), (transRHS ? rhs.columns() : rhs.rows()));

    const auto k = transLHS ? lhs.rows() : lhs.columns();

    const CBLAS_TRANSPOSE tlhs = transLHS ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE trhs = transRHS ? CblasTrans : CblasNoTrans;

    auto blas_multiplication = [&](const BlockRange& r) {
        assert_ge(r.size, (point_type{0, 0}));

        const point_type l_start = transLHS ? point_type{0, r.start.x} : point_type{r.start.x, 0};
        const point_type r_start = transRHS ? point_type{r.start.y, 0} : point_type{0, r.start.y};

        f(CblasRowMajor, tlhs, trhs, r.size.x, r.size.y, k, 1.0, &lhs[l_start], lhs.columns(), &rhs[r_start], rhs.columns(), 0.0,
          &result[{r.start.x, r.start.y}], result.columns());
    };

    #ifdef BLA_MM_PARALLEL_CUTOFF
        const coordinate_type cutoff = BLA_MM_PARALLEL_CUTOFF * BLA_MM_PARALLEL_CUTOFF;
    #else
        const coordinate_type cutoff = std::max((coordinate_type)256, result.rows() * result.columns() / 256);
    #endif

    auto multiplication_rec = allscale::api::core::prec(
        // base case test
        [&](const BlockRange& r) { return r.area() <= cutoff; },
        // base case
        blas_multiplication,
        allscale::api::core::pick(
            // parallel recursive split
            [&](const BlockRange& r, const auto& rec) {
                auto mid = r.start + r.size / 2;

                BlockRange top_left{r.start, mid - r.start};
                BlockRange top_right{{r.start.x, mid.y}, {mid.x - r.start.x, r.start.y + r.size.y - mid.y}};
                BlockRange bottom_left{{mid.x, r.start.y}, {r.start.x + r.size.x - mid.x, mid.y - r.start.y}};
                BlockRange bottom_right{mid, r.start + r.size - mid};

                return allscale::api::core::parallel(allscale::api::core::parallel(rec(top_left), rec(top_right)), allscale::api::core::parallel(rec(bottom_left), rec(bottom_right)));
            },
            // BLAS multiplication if no further parallelism can be exploited
            [&](const BlockRange& r, const auto&) {
                blas_multiplication(r);
                return allscale::api::core::done();
            }));

    multiplication_rec(BlockRange{{0, 0}, result.size()}).wait();
}

// -- parallel blocked blas using pointers
template <bool transLHS = false, bool transRHS = false, typename T, typename Func>
void matrix_multiplication_pbblas(T* result, const T* lhs, const T* rhs, Func f, coordinate_type m, coordinate_type n, coordinate_type k, coordinate_type lda,
                                  coordinate_type ldb, coordinate_type ldc) {
    //    assert_eq((transLHS ? lhs.rows() : lhs.columns()), (transRHS ? rhs.columns() : rhs.rows()));

    //    const auto k = transLHS ? lhs.rows() : lhs.columns();

    //    std::cout << m << " / " << n << " / " << k << std::endl;

    const CBLAS_TRANSPOSE tlhs = transLHS ? CblasTrans : CblasNoTrans;
    const CBLAS_TRANSPOSE trhs = transRHS ? CblasTrans : CblasNoTrans;

    auto blas_multiplication = [&](const BlockRange& r) {
        assert_ge(r.size, (point_type{0, 0}));

        const point_type l_start = transLHS ? point_type{0, r.start.x} : point_type{r.start.x, 0};
        const point_type r_start = transRHS ? point_type{r.start.y, 0} : point_type{0, r.start.y};

        f(CblasRowMajor, tlhs, trhs, r.size.x, r.size.y, k, 1.0, lhs + l_start.x * lda + l_start.y, lda, rhs + r_start.x * ldb + r_start.y, ldb, 0.0,
          result + r.start.x * ldc + r.start.y, ldc);
    };

    #ifdef BLA_MM_PARALLEL_CUTOFF
        const coordinate_type cutoff = BLA_MM_PARALLEL_CUTOFF * BLA_MM_PARALLEL_CUTOFF;
    #else
        const coordinate_type cutoff = std::max((coordinate_type)256, m * n / 256);
    #endif

    auto multiplication_rec = allscale::api::core::prec(
        // base case test
        [&](const BlockRange& r) { return r.area() <= cutoff; },
        // base case
        blas_multiplication,
        allscale::api::core::pick(
            // parallel recursive split
            [&](const BlockRange& r, const auto& rec) {
                auto mid = r.start + r.size / 2;

                BlockRange top_left{r.start, mid - r.start};
                BlockRange top_right{{r.start.x, mid.y}, {mid.x - r.start.x, r.start.y + r.size.y - mid.y}};
                BlockRange bottom_left{{mid.x, r.start.y}, {r.start.x + r.size.x - mid.x, mid.y - r.start.y}};
                BlockRange bottom_right{mid, r.start + r.size - mid};

                return allscale::api::core::parallel(allscale::api::core::parallel(rec(top_left), rec(top_right)), allscale::api::core::parallel(rec(bottom_left), rec(bottom_right)));
            },
            // BLAS multiplication if no further parallelism can be exploited
            [&](const BlockRange& r, const auto&) {
                blas_multiplication(r);
                return allscale::api::core::done();
            }));

    multiplication_rec(BlockRange{{0, 0}, {m, n}}).wait();
}

// -- Strassen-Winograd's matrix multiplication algorithm

template <int Depth = 2048, typename T>
Matrix<T> strassen(const Matrix<T>& A, const Matrix<T>& B) {
    assert_eq(A.columns(), B.rows());

    auto max = std::max({A.columns(), A.rows(), B.columns(), B.rows()});
    long m = std::pow(2, int(std::ceil(std::log2(max))));

    point_type size{m, m};

    if(A.size() == size && B.size() == size) {
        // no need to resize
        Matrix<T> result(size);

        // call the recursive strassen algorithm
        detail::strassen_rec<Depth>(SubMatrix<const Matrix<T>>(A), SubMatrix<const Matrix<T>>(B), SubMatrix<Matrix<T>>(result));

        return result;
    } else {
        // resize and call the actual strassen algorithm
        Matrix<T> A_padded(size);
        Matrix<T> B_padded(size);

        A_padded.sub({{0, 0}, A.size()}) = A;
        B_padded.sub({{0, 0}, B.size()}) = B;

        Matrix<T> result_padded(size);

        // call the recursive strassen algorithm
        detail::strassen_rec<Depth>(SubMatrix<const Matrix<T>>(A_padded), SubMatrix<const Matrix<T>>(B_padded), SubMatrix<Matrix<T>>(result_padded));

        Matrix<T> result({A.rows(), B.columns()});

        // write the padded result to the actual result matrix
        result = result_padded.sub({{0, 0}, result.size()});

        return result;
    }
}

// -- default column vector * row vector multiplication
template <typename T, typename E1, typename E2>
void vv_multiplication(Matrix<T>& result, const MatrixExpression<E1>& lhs, const MatrixExpression<E2>& rhs) {
    assert_eq(lhs.columns(), 1);
    assert_eq(rhs.rows(), 1);

    using vt = Vc::Vector<T>;


    const int packet_size = vt::size();
    const int caligned_end = result.columns() / packet_size;

    // calculate using vectorization
    allscale::api::user::algorithm::pfor(point_type{result.rows(), caligned_end}, [&](const auto& vec_pos) {
        point_type pos(vec_pos.x, vec_pos.y * packet_size);

        vt left(lhs[{pos.x, 0}]);

        vt right(&static_cast<const E2&>(rhs)[{0, pos.y}]);

        (left * right).store(&result[pos]);
    });

    // calculate rest
    allscale::api::user::algorithm::pfor(point_type{0, caligned_end * packet_size}, result.size(), [&](const auto& pos) { result[pos] = lhs[{pos.x, 0}] * rhs[{0, pos.y}]; });
}

// -- default matrix * matrix multiplication
template <typename T, typename E1, typename E2>
std::enable_if_t<!(direct_or_transpose_v<E1> && direct_or_transpose_v<E2>)> matrix_multiplication(Matrix<T>& result, const MatrixExpression<E1>& lhs,
                                                                                                  const MatrixExpression<E2>& rhs) {
    matrix_multiplication(result, lhs.eval(), rhs.eval());
}

// -- row permutation
template <typename T, typename E1, typename E2>
void matrix_multiplication(Matrix<T>& result, const PermutationMatrix<E1>& lhs, const MatrixExpression<E2>& rhs) {
    assert_eq(lhs.columns(), rhs.rows());

    allscale::api::user::algorithm::pfor(allscale::utils::Vector<coordinate_type, 1>(result.rows()), [&](const auto& pos) {
        const coordinate_type i = pos[0];
        detail::evaluate_simplify(rhs.row(lhs.permutation(i)), result.row(i));
    });
}

// -- column permutation
template <typename T, typename E1, typename E2>
void matrix_multiplication(Matrix<T>& result, const MatrixExpression<E1>& lhs, const MatrixTranspose<PermutationMatrix<E2>>& rhs) {
    assert_eq(lhs.columns(), rhs.rows());

    allscale::api::user::algorithm::pfor(result.size(), [&](const auto& pos) { result[pos] = lhs[{pos.x, rhs.getExpression().permutation(pos.y)}]; });
}

template <typename T>
std::enable_if_t<!std::is_same<double, T>::value && !std::is_same<float, T>::value> matrix_multiplication(Matrix<T>& result, const Matrix<T>& lhs,
                                                                                                          const Matrix<T>& rhs) {
    allscale::api::user::algorithm::pfor(result.size(), [&](const point_type& p) {
        result[p] = 0;
        for(int i = 0; i < lhs.columns(); ++i) {
            result[p] += lhs[{p.x, i}] * rhs[{i, p.y}];
        }
    });
}

// -- double
template <typename E1, typename E2>
std::enable_if_t<direct_or_transpose_v<E1> && direct_or_transpose_v<E2>> matrix_multiplication(Matrix<double>& result, const MatrixExpression<E1>& lhs,
                                                                                               const MatrixExpression<E2>& rhs) {
    // TODO: make a compile time fix and float
    if(lhs.columns() == 1 && rhs.rows() == 1) {
        vv_multiplication(result, lhs, rhs);
        return;
    }
    matrix_multiplication_pbblas<is_transpose_v<E1>, is_transpose_v<E2>>(
        &result[{0, 0}], &static_cast<const E1&>(lhs)[{0, 0}], &static_cast<const E2&>(rhs)[{0, 0}], cblas_dgemm, result.rows(), result.columns(),
        lhs.columns(), static_cast<const E1&>(lhs).stride(), static_cast<const E2&>(rhs).stride(), result.stride());
}

// -- float
template <typename E1, typename E2>
std::enable_if_t<direct_or_transpose_v<E1> && direct_or_transpose_v<E2>> matrix_multiplication(Matrix<float>& result, const MatrixExpression<E1>& lhs,
                                                                                               const MatrixExpression<E2>& rhs) {
    matrix_multiplication_pbblas<is_transpose_v<E1>, is_transpose_v<E2>>(
        &result[{0, 0}], &static_cast<const E1&>(lhs)[{0, 0}], &static_cast<const E2&>(rhs)[{0, 0}], cblas_sgemm, result.rows(), result.columns(),
        lhs.columns(), static_cast<const E1&>(lhs).stride(), static_cast<const E2&>(rhs).stride(), result.stride());
}

} // end namespace impl
} // namespace bla
