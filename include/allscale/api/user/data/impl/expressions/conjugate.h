#pragma once

#include "allscale/api/user/data/impl/expressions/expression.h"
#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/traits.h"
#include "allscale/api/user/data/impl/types.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename E>
class MatrixConjugate : public MatrixExpression<MatrixConjugate<E>> {
    using typename MatrixExpression<MatrixConjugate<E>>::T;

    using Exp = expression_member_t<E>;

public:
    MatrixConjugate(Exp u) : expression(u) {
    }

    T operator[](const point_type& pos) const {
        return detail::conj(expression[pos]);
    }

    point_type size() const {
        return expression.size();
    }

    coordinate_type rows() const {
        return expression.rows();
    }

    coordinate_type columns() const {
        return expression.columns();
    }

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
