#pragma once

#include "allscale/api/user/data/impl/expressions/expression.h"
#include "allscale/api/user/data/impl/expressions/identity.h"
#include "allscale/api/user/data/impl/types.h"

#include <allscale/api/user/data/grid.h>
#include <allscale/api/user/data/matrix.h>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename T>
struct EigenSolver {
    EigenSolver(const Matrix<T>& matrix) {
        assert_eq(matrix.rows(), matrix.columns());
        compute(matrix);
    }

private:
    bool isUpper(const Matrix<T>& m) {
        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < i; ++j) {
                // TODO: epsilon
                if(m[{i, j}] > 1E-8) {
                    return false;
                }
            }
        }

        return true;
    }

    void compute(const Matrix<T>& matrix) {
        using ct = coordinate_type;

        Matrix<T> m = matrix;
        Matrix<T> vec(IdentityMatrix<T>(matrix.size()));

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

public:
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
