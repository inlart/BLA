#pragma once

#include <bla/matrix.h>

namespace bla {
namespace impl {

template <typename E1, typename E2>
Matrix<scalar_type_t<E1>> gemm(const MatrixExpression<E1>& e1, const MatrixExpression<E2>& e2) {
    assert_eq(e1.columns(), e2.rows());
    Matrix<scalar_type_t<E1>> result({e1.rows(), e2.columns()});
    for(int i = 0; i < result.rows(); ++i) {
        for(int j = 0; j < result.columns(); ++j) {
            for(int k = 0; k < e1.columns(); ++k) {
                result[{i, j}] += e1[{i, k}] * e2[{k, j}];
            }
        }
    }

    return result;
}

} // namespace impl
} // namespace bla
