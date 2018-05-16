#pragma once

#include <allscale/api/user/data/matrix.h>

template <typename E>
using MatrixExpression = allscale::api::user::data::MatrixExpression<E>;

template <typename E1, typename E2>
bool isAlmostEqual(const MatrixExpression<E1>& a, const MatrixExpression<E2>& b, double epsilon = 0.001) {
    if(a.size()[0] != b.size()[0] || a.size()[1] != b.size()[1]) {
        return false;
    }
    for(allscale::api::user::data::coordinate_type i = 0; i < a.rows(); ++i) {
        for(allscale::api::user::data::coordinate_type j = 0; j < a.columns(); ++j) {
            double diff = (a[{i, j}] - b[{i, j}]);
            if(diff * diff > epsilon) {
                return false;
            }
        }
    }
    return true;
}
