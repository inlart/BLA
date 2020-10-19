#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/traits.h"
#include "bla/impl/types.h"

namespace bla {
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
} // namespace bla
