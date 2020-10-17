#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/expressions/identity.h"
#include "bla/impl/types.h"

#include "bla/matrix.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename T>
struct EigenSolver {
    EigenSolver(const Matrix<T>& matrix) : Q(IdentityMatrix<T>(matrix.size())) {
        assert_eq(matrix.rows(), matrix.columns());
        Matrix<T> H = matrix;
        hessenberg_form(H);
        compute(H);
    }

    T getEigenvalue(coordinate_type i) {
        return eigenvalues[i];
    }

    SubMatrix<const Matrix<T>> getEigenvector(coordinate_type i) {
        return Q.column(i);
    }

private:
    bool isUpper(const Matrix<T>& m) {
        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < i; ++j) {
                // TODO: epsilon
                if(std::abs(m[{i, j}]) > 1E-9) {
                    return false;
                }
            }
        }

        return true;
    }

    void compute(const Matrix<T>& matrix) {
        using ct = coordinate_type;

        Matrix<T> m = matrix;
        Matrix<T> vec(Q);

        // -- QR Algorithm
        // stop when the matrix is upper triangular
        while(!isUpper(m)) {
            // perform QR decomposition
            auto qr = m.QRDecomposition();

            // calculate new matrix = R * Q
            m = qr.getR() * qr.getQ();

            // update eigenvector computation
            Q *= qr.getQ();
        }


        for(ct i = 0; i < m.rows(); ++i) {
            // eigenvalues are on the diagonal
            eigenvalues.push_back(m[{i, i}]);

            // eigenvectors are already in Q
        }
    }

    void hessenberg_form(Matrix<T>& H) {
        // bring matrix to upper hessenberg form
        for(int k = 0; k < H.rows() - 2; ++k) {
            Householder<T> h(H.column(k).bottomRows(H.rows() - k - 1), H.size());

            Q *= h.getP();

            h.applyLeft(H);
            h.applyRight(H);
        }
    }

public:
    Matrix<T> Q;
    std::vector<T> eigenvalues;
};

template <typename E>
EigenSolver<scalar_type_t<E>> MatrixExpression<E>::solveEigen() const {
    return EigenSolver<scalar_type_t<E>>(static_cast<const E&>(*this));
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
