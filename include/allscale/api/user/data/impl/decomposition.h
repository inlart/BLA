#pragma once

#include "expressions.h"
#include "traits.h"

#include "forward.h"

#include <allscale/api/user/data/grid.h>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

using point_type = GridPoint<2>;

template <typename T>
struct LUD {
	LUD(const Matrix<T>& A) : L(A.size()), U(A.size()) {
		using ct = coordinate_type;
		assert_eq(A.rows(), A.columns());

		ct n = A.rows();

		for(ct i = 0; i < n; ++i) {
			for(ct j = 0; j < n; ++j) {
				if(j < i) {
					L[{j, i}] = static_cast<T>(0);
				} else {
					L[{j, i}] = A[{j, i}];
					for(ct k = 0; k < i; ++k) {
						L[{j, i}] -= L[{j, k}] * U[{k, i}];
					}
				}
			}
			for(ct j = 0; j < n; ++j) {
				if(j < i) {
					U[{i, j}] = static_cast<T>(0);
				} else if(j == i) {
					U[{i, j}] = static_cast<T>(1);
				} else {
					U[{i, j}] = A[{i, j}] / L[{i, i}];
					for(ct k = 0; k < i; ++k) {
						U[{i, j}] -= L[{i, k}] * U[{k, j}] / L[{i, i}];
					}
				}
			}
		}
	}

	LUD(const LUD<T>&) = delete;
	LUD(LUD<T>&&) = default;

	LUD<T>& operator=(const LUD<T>&) = delete;
	LUD<T>& operator=(LUD<T>&&) = default;


	const Matrix<T>& lower() const { return L; }

	const Matrix<T>& upper() { return U; }

  private:
	Matrix<T> L;
	Matrix<T> U;
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
