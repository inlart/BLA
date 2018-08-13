#pragma once

#include "allscale/api/user/data/impl/expressions/expression.h"
#include "allscale/api/user/data/impl/expressions/identity.h"
#include "allscale/api/user/data/impl/types.h"

#include <allscale/api/user/data/matrix.h>

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

        // QR Algorithm
        while(!isUpper(m)) {
            auto qr = m.QRDecomposition();
            m = qr.getR() * qr.getQ();
            vec *= qr.getQ();
        }


        for(ct i = 0; i < m.rows(); ++i) {
            // eigenvalues are on the diagonal
            eigenvalues.push_back(m[{i, i}]);

            eigenvectors.push_back((vec.column(i)).eval());
        }
    }

    void hessenberg_form(Matrix<T>& H) {
        for(int k = 0; k < H.rows() - 2; ++k) {
            Matrix<T> old = H;
            Householder<T> h({H.column(k).bottomRows(H.rows() - k - 1)}, H.size());

            Q *= h.getP();

            h.applyLeft(H);
            h.applyRight(H);
        }
    }

public:
    Matrix<T> Q;
    std::vector<T> eigenvalues;
    std::vector<Matrix<T>> eigenvectors;
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
