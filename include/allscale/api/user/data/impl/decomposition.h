#pragma once

#include "expressions.h"
#include "traits.h"

#include "forward.h"

#include <allscale/api/user/data/grid.h>
#include <allscale/utils/assert.h>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

using point_type = GridPoint<2>;

// -- partial pivoting LU decomposition with PA = LU
template <typename T>
struct LUD {
    LUD(const Matrix<T>& A) : P(A.rows()), LU(A) {
        using ct = coordinate_type;
        assert_eq(A.rows(), A.columns());

        const T epsilon = static_cast<T>(1E-4);

        // -- compute permutation matrix
        for(ct i = 0; i < A.rows(); ++i) {
            T max_value = static_cast<T>(0);
            ct max_column = i;

            for(ct k = i; k < A.rows(); ++k) {
                T value = std::abs(LU[{k, i}]);
                if(value > max_value) {
                    max_value = value;
                    max_column = k;
                }
            }

            if(max_value < epsilon) {
                assert_fail();
                return;
            }

            if(max_column != i) {
                P.swap(i, max_column);
                LU.row(i).swap(LU.row(max_column));
            }

            for(ct j = i + 1; j < A.rows(); ++j) {
                LU[{j, i}] /= LU[{i, i}];

                for(ct k = i + 1; k < A.rows(); ++k) {
                    LU[{j, k}] -= LU[{j, i}] * LU[{i, k}];
                }
            }
        }
    }

    LUD(const LUD<T>&) = delete;
    LUD(LUD<T>&&) = default;

    LUD<T>& operator=(const LUD<T>&) = delete;
    LUD<T>& operator=(LUD<T>&&) = default;


    Matrix<T> lower() const {
        Matrix<T> l(LU.size());
        l.fill([&](const auto& pos){
            if(pos.x > pos.y) {
                return LU[pos];
            }
            else if(pos.x == pos.y) {
                return static_cast<T>(1);
            }
            else {
                return static_cast<T>(0);
            }
        });

        return l;
    }

    Matrix<T> upper() const {
        Matrix<T> u(LU.size());
        u.fill([&](const auto& pos){
            if(pos.x <= pos.y) {
                return LU[pos];
            }
            else {
                return static_cast<T>(0);
            }
        });
        return u;
    }

    const PermutationMatrix<T>& permutation() const { return P; }

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
                }
                else {
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

			if(mag < 1E-10) continue;

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

	const Matrix<T>& getQ() const { return Q; }
	const Matrix<T>& getR() const { return R; }

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

	const Matrix<T>& getU() { return U; }
	const Matrix<T>& getS() { return S; }
	const Matrix<T>& getV() { return V; }

  private:
	Matrix<T> U;
	Matrix<T> S;
	Matrix<T> V;
};


} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
