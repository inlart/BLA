#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/types.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename T>
class IdentityMatrix : public MatrixExpression<IdentityMatrix<T>> {
public:
    IdentityMatrix(point_type matrix_size) : matrix_size(matrix_size) {
    }

    T operator[](const point_type& pos) const {
        assert_lt(pos, matrix_size);
        return pos.x == pos.y ? static_cast<T>(1) : static_cast<T>(0);
    }

    point_type size() const {
        return matrix_size;
    }

    coordinate_type rows() const {
        return matrix_size[0];
    }

    coordinate_type columns() const {
        return matrix_size[1];
    }

private:
    point_type matrix_size;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
