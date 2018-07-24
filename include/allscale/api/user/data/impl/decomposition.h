#pragma once

#include <algorithm>
#include <allscale/api/user/data/grid.h>
#include <allscale/utils/assert.h>
#include <limits>

#include "allscale/api/user/data/impl/expressions.h"
#include "allscale/api/user/data/impl/forward.h"
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
        compute_blocked(LU, t);
        P = t;
    }

    LUD(const LUD<T>&) = delete;
    LUD(LUD<T>&&) = default;

    LUD<T>& operator=(const LUD<T>&) = delete;
    LUD<T>& operator=(LUD<T>&&) = default;


    Matrix<T> lower() const {
        Matrix<T> l(LU.size());
        l.fill([&](const auto& pos) {
            if(pos.x > pos.y) {
                return LU[pos];
            } else if(pos.x == pos.y) {
                return static_cast<T>(1);
            } else {
                return static_cast<T>(0);
            }
        });

        return l;
    }

    Matrix<T> upper() const {
        Matrix<T> u(LU.size());
        u.fill([&](const auto& pos) {
            if(pos.x <= pos.y) {
                return LU[pos];
            } else {
                return static_cast<T>(0);
            }
        });
        return u;
    }

    const PermutationMatrix<T>& permutation() const {
        return P;
    }

    T determinant() const {
        using ct = coordinate_type;

        T det = LU[{0, 0}];

        const ct n = LU.rows();

        for(ct i = 1; i < n; ++i) {
            det *= LU[{i, i}];
        }

        if((P.numSwaps() & 1) == 0)
            return det;
        else
            return -det;
    }

    Matrix<T> inverse() const {
        using ct = coordinate_type;
        Matrix<T> inverse(LU.size());


        for(ct j = 0; j < LU.rows(); ++j) {
            for(ct i = 0; i < LU.rows(); ++i) {
                if(P.permutation(i) == j) {
                    inverse[{i, j}] = static_cast<T>(1);
                } else {
                    inverse[{i, j}] = static_cast<T>(0);
                }

                for(ct k = 0; k < i; ++k) {
                    inverse[{i, j}] -= LU[{i, k}] * inverse[{k, j}];
                }
            }

            for(ct i = LU.rows() - 1; i >= 0; --i) {
                for(ct k = i + 1; k < LU.rows(); ++k) {
                    inverse[{i, j}] -= LU[{i, k}] * inverse[{k, j}];
                }
                inverse[{i, j}] = inverse[{i, j}] / LU[{i, i}];
            }
        }

        return inverse;
    }

    // -- Solve A * x = b for x
    Matrix<T> solve(SubMatrix<Matrix<T>> b) {
        assert_eq(b.rows(), LU.columns());
        using ct = coordinate_type;
        Matrix<T> x(b.size());

        // TODO: pfor
        for(ct ii = 0; ii < x.columns(); ++ii) {
            for(ct i = 0; i < LU.rows(); ++i) {
                x[{i, ii}] = b[{P.permutation(i), ii}];

                for(ct k = 0; k < i; ++k) {
                    x[{i, ii}] -= LU[{i, k}] * x[{k, ii}];
                }
            }
            for(ct i = LU.rows() - 1; i >= 0; --i) {
                for(ct k = i + 1; k < LU.rows(); ++k) {
                    x[{i, ii}] -= LU[{i, k}] * x[{k, ii}];
                }

                x[{i, ii}] /= LU[{i, i}];
            }
        }

        return x;
    }

private:
    void compute_unblocked(SubMatrix<Matrix<T>> loup, Transpositions& t) {
        using ct = coordinate_type;

        const ct size = std::min(loup.rows(), loup.columns());

        const ct start_row = loup.getBlockRange().start.x;


        for(ct k = 0; k < size; ++k) {
            auto it = loup.column(k).bottomRows(loup.rows() - k).abs().max_element();

            ct max_row = it - loup.begin();

            max_row += k;
            t[k + start_row] = max_row + start_row;
            if(*it != 0) {
                if(k != max_row) {
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

    void compute_blocked(SubMatrix<Matrix<T>> loup, Transpositions& t) {
        using ct = coordinate_type;

        const ct maxBlockSize = 256;

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


            compute_blocked(loup.sub({{k, k}, {trows + bs, bs}}), t);

            // update permutations and apply them to A_0
            for(ct i = k; i < k + bs; ++i) {
                A_0.row(i).swap(A_0.row(t[i + loup.getBlockRange().start.x] - loup.getBlockRange().start.x));
            }

            if(trows) {
                // apply permutations to A_2
                for(ct i = k; i < k + bs; ++i) {
                    A_2.row(i).swap(A_2.row(t[i + loup.getBlockRange().start.x] - loup.getBlockRange().start.x));
                }

                // TODO: improve this
                auto x = A11.template view<ViewType::UnitLower>().LUDecomposition();
                A12 = x.solve(A12);
                // A12 = A11^-1 A12
                //                A11.template triangularView<UnitLower>().solveInPlace(A12);
                A22 -= A21 * A12;
            }
        }
    }

private:
    PermutationMatrix<T> P;
    Matrix<T> LU;
};

template <typename T>
struct QRD {
    QRD(const Matrix<T>& A) : Q(point_type{A.rows(), A.rows()}), R(A) {
        using ct = coordinate_type;

        // Householder QR Decomposition
        T mag, alpha;
        Matrix<T> u({A.rows(), 1});
        Matrix<T> v({A.rows(), 1});

        Matrix<T> P(point_type{A.rows(), A.rows()});
        Matrix<T> I(IdentityMatrix<T>(point_type{A.rows(), A.rows()})); // TODO: fix

        Q.identity();

        for(ct i = 0; i < A.columns(); ++i) {
            u.zero();
            v.zero();

            mag = 0;
            for(ct j = i; j < A.rows(); ++j) {
                u[{j, 0}] = R[{j, i}];
                mag += u[{j, 0}] * u[{j, 0}];
            }
            mag = std::sqrt(mag);

            alpha = u[{i, 0}] < 0 ? mag : -mag;

            mag = 0.0;
            for(ct j = i; j < A.rows(); ++j) {
                v[{j, 0}] = j == i ? u[{j, 0}] + alpha : u[{j, 0}];

                mag += v[{j, 0}] * v[{j, 0}];
            }
            mag = std::sqrt(mag);

            if(mag < 1E-10)
                continue;

            for(ct j = i; j < A.rows(); ++j) {
                v[{j, 0}] /= mag;
            }

            P = I - (v * v.transpose()) * 2.0;

            R = (P * R).eval();
            Q *= P;
        }
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
