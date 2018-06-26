#pragma once

#include <allscale/api/user/data/grid.h>
#include <allscale/utils/assert.h>

#include "allscale/api/user/data/impl/expressions.h"
#include "allscale/api/user/data/impl/traits.h"

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

        lu_unblocked();
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
    Matrix<T> solve(const Matrix<T>& b) {
        assert_eq(b.size(), (point_type{LU.rows(), 1}));
        using ct = coordinate_type;
        Matrix<T> x(b.size());

        for(ct i = 0; i < LU.rows(); ++i) {
            x[{i, 0}] = b[{P.permutation(i), 0}];

            for(ct k = 0; k < i; ++k) {
                x[{i, 0}] -= LU[{i, k}] * x[{k, 0}];
            }
        }

        for(ct i = LU.rows() - 1; i >= 0; --i) {
            for(ct k = i + 1; k < LU.rows(); ++k) {
                x[{i, 0}] -= LU[{i, k}] * x[{k, 0}];
            }

            x[{i, 0}] /= LU[{i, i}];
        }

        return x;
    }

private:
    void lu_unblocked() {
        using ct = coordinate_type;

        for(ct k = 0; k < LU.rows(); ++k) {
            auto it = LU.column(k).bottomRows(LU.rows() - k).abs().max_element();
            ct max_row = (it - LU.begin()) / LU.columns();
            max_row += k;

            P.swap(k, max_row);
            if(*it != 0) {
                if(k != it - LU.begin()) {
                    LU.row(k).swap(LU.row(max_row));
                }

                LU.column(k).bottomRows(LU.rows() - k - 1) /= LU[{k, k}];
            }
            if(k < LU.rows() - 1) {
                LU.bottomRows(LU.rows() - k - 1).bottomColumns(LU.columns() - k - 1) -=
                    LU.column(k).bottomRows(LU.rows() - k - 1) * LU.row(k).bottomColumns(LU.columns() - k - 1);
            }
        }
    }

    void lu_blocked() {
        // TODO: implemenet blocked LU decomposition
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

            R = P * R;
            Q = Q * P;
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


} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
