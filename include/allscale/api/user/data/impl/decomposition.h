#pragma once

#include <algorithm>
#include <allscale/api/user/data/grid.h>
#include <allscale/utils/assert.h>
#include <limits>

#include "allscale/api/user/data/impl/expressions.h"
#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/householder.h"
#include "allscale/api/user/data/impl/traits.h"
#include "allscale/api/user/data/impl/transpositions.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

using point_type = GridPoint<2>;

// -- partial pivoting LU decomposition with PA = LU
template <typename T>
struct LUD {
    static_assert(!std::numeric_limits<T>::is_integer, "Decomposition only for floating point types");

    LUD(const Matrix<T>& A) : P(A.rows()), LU(A) {
        assert_eq(A.rows(), A.columns());
        Transpositions t(LU.columns());
        compute_blocked(SubMatrix<Matrix<T>>(LU), t);
        P = t;
    }

    LUD(const LUD<T>&) = delete;
    LUD(LUD<T>&&) = default;

    LUD<T>& operator=(const LUD<T>&) = delete;
    LUD<T>& operator=(LUD<T>&&) = default;


    MatrixView<const Matrix<T>, ViewType::UnitLower> lower() const {
        return LU.template view<ViewType::UnitLower>();
    }

    MatrixView<const Matrix<T>, ViewType::Upper> upper() const {
        return LU.template view<ViewType::Upper>();
    }

    const PermutationMatrix<T>& permutation() const {
        return P;
    }

    T determinant() const {
        using ct = coordinate_type;

        T det = LU[{0, 0}];

        const ct n = LU.rows();

        // determinant is the multiplication of
        // the diagonal elements of the lower and upper matrix
        for(ct i = 1; i < n; ++i) {
            det *= LU[{i, i}];
        }

        if((P.numSwaps() & 1) == 0)
            return det;
        else
            return -det;
    }

    Matrix<T> inverse() const {
        // A = P^T * L * U
        // --> A^-1 = U^-1 * L^-1 * P
        Matrix<T> inverse(upper().inverse());
        inverse *= lower().inverse();
        inverse *= permutation();

        return inverse;
    }

    // -- Solve A * x = b for x
    Matrix<T> solve(SubMatrix<Matrix<T>> b) {
        // L * U * x = P * b
        assert_eq(b.rows(), LU.columns());
        Matrix<T> x(b.size());

        x = P * b;

        // L * U * x' = x
        LU.template view<ViewType::UnitLower>().solveInPlace(x);

        // U * x' = x
        LU.template view<ViewType::Upper>().solveInPlace(x);

        return x;
    }

private:
    void compute_unblocked(SubMatrix<Matrix<T>> loup, Transpositions& t) {
        using ct = coordinate_type;

        const ct size = std::min(loup.rows(), loup.columns());

        const ct start_row = loup.getBlockRange().start.x;


        for(ct k = 0; k < size; ++k) {
            auto abs_range = loup.column(k).bottomRows(loup.rows() - k).abs();

            // find the row of the max element
            auto it = abs_range.max_element();
            ct max_row = it - abs_range.begin();
            max_row += k;

            // add switch to transpositions
            t[k + start_row] = max_row + start_row;

            if(*it != 0) {
                if(k != max_row) {
                    // swap the rows
                    loup.row(k).swap(loup.row(max_row));
                }

                loup.column(k).bottomRows(loup.rows() - k - 1) /= loup[{k, k}];
            }
            if(k < loup.rows() - 1) {
                loup.bottomRows(loup.rows() - k - 1).bottomColumns(loup.columns() - k - 1) -=
                    loup.column(k).bottomRows(loup.rows() - k - 1) * loup.row(k).bottomColumns(loup.columns() - k - 1);
            }
        }
    }

    void compute_blocked(SubMatrix<Matrix<T>> loup, Transpositions& t, const coordinate_type maxBlockSize = 256) {
        using ct = coordinate_type;

        const ct size = std::min(loup.rows(), loup.columns());
        const ct rows = loup.rows();

        if(size <= 16) {
            compute_unblocked(loup, t);
            return;
        }

        ct blockSize;
        blockSize = size / 8;
        blockSize = (blockSize / 16) * 16;
        blockSize = std::min(std::max(blockSize, ct(8)), maxBlockSize);

        for(ct k = 0; k < size; k += blockSize) {
            ct bs = std::min(size - k, blockSize); // actual size of the block
            ct trows = rows - k - bs;              // trailing rows
            ct tsize = size - k - bs;              // trailing size

            // partition the matrix:
            //                          A00 | A01 | A02
            // lu  = A_0 | A_1 | A_2 =  A10 | A11 | A12
            //                          A20 | A21 | A22
            auto A_0 = loup.sub({{0, 0}, {rows, k}});
            auto A_2 = loup.sub({{0, k + bs}, {rows, tsize}});
            auto A11 = loup.sub({{k, k}, {bs, bs}});
            auto A12 = loup.sub({{k, k + bs}, {bs, tsize}});
            auto A21 = loup.sub({{k + bs, k}, {trows, bs}});
            auto A22 = loup.sub({{k + bs, k + bs}, {trows, tsize}});


            // call recursive compute
            compute_blocked(loup.sub({{k, k}, {trows + bs, bs}}), t, 16);

            // update permutations and apply them to A_0
            for(ct i = k; i < k + bs; ++i) {
                A_0.row(i).swap(A_0.row(t[i + loup.getBlockRange().start.x] - loup.getBlockRange().start.x));
            }

            if(trows) {
                // apply permutations to A_2
                for(ct i = k; i < k + bs; ++i) {
                    A_2.row(i).swap(A_2.row(t[i + loup.getBlockRange().start.x] - loup.getBlockRange().start.x));
                }

                A11.template view<ViewType::UnitLower>().solveInPlace(A12);

                A22 -= A21 * A12;
            }
        }
    }

private:
    PermutationMatrix<T> P;
    Matrix<T> LU;
};

// -- full pivoting LUD
template <typename T>
struct FPLUD {
    static_assert(!std::numeric_limits<T>::is_integer, "Decomposition only for floating point types");

    FPLUD(const Matrix<T>& A) : P(A.rows()), Q(A.columns()), LU(A) {
        assert_eq(A.rows(), A.columns());
        compute_unblocked(LU);
    }

    FPLUD(const FPLUD<T>&) = delete;
    FPLUD(FPLUD<T>&&) = default;

    FPLUD<T>& operator=(const FPLUD<T>&) = delete;
    FPLUD<T>& operator=(FPLUD<T>&&) = default;


    MatrixView<const Matrix<T>, ViewType::UnitLower> lower() const {
        return LU.template view<ViewType::UnitLower>();
    }

    MatrixView<const Matrix<T>, ViewType::Upper> upper() const {
        return LU.template view<ViewType::Upper>();
    }

    const PermutationMatrix<T>& rowPermutation() const {
        return P;
    }

    MatrixTranspose<PermutationMatrix<T>> columnPermutation() const {
        return Q.transpose();
    }

    Matrix<T> inverse() const {
        // A = P^T * L * U * Q^T
        // --> A^-1 = Q * U^-1 * L^-1 * P
        Matrix<T> inverse(columnPermutation());
        inverse *= upper().inverse();
        inverse *= lower().inverse();
        inverse *= rowPermutation();

        return inverse;
    }

    T determinant() const {
        using ct = coordinate_type;

        T det = LU[{0, 0}];

        const ct n = LU.rows();

        // determinant is the multiplication of
        // the diagonal elements of the lower and upper matrix
        for(ct i = 1; i < n; ++i) {
            det *= LU[{i, i}];
        }

        if(((P.numSwaps() + Q.numSwaps()) & 1) == 0)
            return det;
        else
            return -det;
    }


    // -- Solve A * x = b for x
    Matrix<T> solve(SubMatrix<Matrix<T>> b) {
        assert_eq(b.rows(), LU.columns());
        Matrix<T> x(b.size());

        x = P * b;

        LU.template view<ViewType::UnitLower>().solveInPlace(x);
        LU.template view<ViewType::Upper>().solveInPlace(x);

        return Q.transpose() * x;
    }

    int rank() const {
        int rank = 0;
        // rank is the number of elements greater than zero
        for(int i = 0; i < LU.rows(); ++i) {
            if(std::abs(LU[{i, i}]) > 1E-8)
                ++rank;
        }
        return rank;
    }


private:
    void compute_unblocked(SubMatrix<Matrix<T>> loup) {
        using ct = coordinate_type;

        const ct size = std::min(loup.rows(), loup.columns());

        for(ct k = 0; k < size; ++k) {
            auto abs_range = loup.bottomColumns(loup.columns() - k).bottomRows(loup.rows() - k).abs();

            // find row and column of max element
            auto it = abs_range.max_element();

            ct max_row = it.pointPos().x;
            ct max_column = it.pointPos().y;

            max_row += k;
            max_column += k;
            if(*it == 0) {
                // TODO: error handling
                return;
            }


            if(k != max_row) {
                // swap row with max element row
                P.swap(k, max_row);
                loup.row(k).swap(loup.row(max_row));
            }

            if(k != max_column) {
                // swap column with max element column
                Q.swap(k, max_column);
                loup.column(k).swap(loup.column(max_column));
            }

            // update bottom right by gaussian elimination
            loup.column(k).bottomRows(loup.rows() - k - 1) /= loup[{k, k}];

            if(k < std::min(loup.rows(), loup.columns()) - 1) {
                loup.bottomRows(loup.rows() - k - 1).bottomColumns(loup.columns() - k - 1) -=
                    loup.column(k).bottomRows(loup.rows() - k - 1) * loup.row(k).bottomColumns(loup.columns() - k - 1);
            }
        }
    }

private:
    PermutationMatrix<T> P;
    PermutationMatrix<T> Q;
    Matrix<T> LU;
};

template <typename T>
struct QRD {
    QRD(const Matrix<T>& A) : Q(IdentityMatrix<T>({A.rows(), A.rows()})), R(A) {
        assert_ge(A.rows(), A.columns());

        compute_unblocked({A});
    }

    QRD(const QRD<T>&) = delete;
    QRD(QRD<T>&&) = default;

    QRD<T>& operator=(const QRD<T>&) = delete;
    QRD<T>& operator=(QRD<T>&&) = default;

    const Matrix<T>& getQ() const {
        return Q;
    }
    const Matrix<T>& getR() const {
        return R;
    }

private:
    void compute_unblocked(SubMatrix<const Matrix<T>> A) {
        using ct = coordinate_type;

        // Householder QR Decomposition
        Q.identity();

        for(ct i = 0; i < A.columns(); ++i) {
            // calculate Householder reflection for i-th column and last rows - i rows
            Householder<T> h(R.column(i).bottomRows(A.rows() - i), Q.size());

            // add rows - i - 1 zeros on i-th column of Matrix R
            h.applyLeft(R);

            // update Q
            h.applyRight(Q);
        }
    }

    Matrix<T> Q;
    Matrix<T> R;
};

template <typename T>
struct SVD {
    SVD(const Matrix<T>& A) : U(point_type{A.rows(), A.rows()}), S(A.size()), V(point_type{A.columns(), A.columns()}) {
        // TODO: implement
        assert_fail();
    }

    SVD(const SVD<T>&) = delete;
    SVD(SVD<T>&&) = default;

    SVD<T>& operator=(const SVD<T>&) = delete;
    SVD<T>& operator=(SVD<T>&&) = default;

    const Matrix<T>& getU() {
        return U;
    }
    const Matrix<T>& getS() {
        return S;
    }
    const Matrix<T>& getV() {
        return V;
    }

private:
    Matrix<T> U;
    Matrix<T> S;
    Matrix<T> V;
};

template <typename E>
LUD<scalar_type_t<E>> MatrixExpression<E>::LUDecomposition() const {
    return LUD<scalar_type_t<E>>(*this);
}

template <typename E>
FPLUD<scalar_type_t<E>> MatrixExpression<E>::FPLUDecomposition() const {
    return FPLUD<scalar_type_t<E>>(*this);
}

template <typename E>
QRD<scalar_type_t<E>> MatrixExpression<E>::QRDecomposition() const {
    return QRD<scalar_type_t<E>>(*this);
}

template <typename E>
SVD<scalar_type_t<E>> MatrixExpression<E>::SVDecomposition() const {
    return SVD<scalar_type_t<E>>(*this);
}

template <typename E>
scalar_type_t<E> MatrixExpression<E>::determinant() const {
    return LUDecomposition().determinant();
}

template <typename E>
Matrix<scalar_type_t<E>> MatrixExpression<E>::inverse() const {
    return LUDecomposition().inverse();
}


} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // end namespace allscale
