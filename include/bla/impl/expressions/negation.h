#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/types.h"

#include <Vc/Vc>

namespace bla {
namespace impl {

template <typename E>
class MatrixNegation : public MatrixExpression<MatrixNegation<E>> {
    using typename MatrixExpression<MatrixNegation<E>>::T;
    using typename MatrixExpression<MatrixNegation<E>>::PacketScalar;

    using Exp = expression_member_t<E>;

public:
    MatrixNegation(Exp e) : expression(e) {
    }
    T operator[](const point_type& pos) const {
        return -expression[pos];
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

    template <typename T, typename simd_type = PacketScalar, typename simd_flags = Vc::UnalignedTag>
    simd_type packet(T p) const {
        return -expression.template packet<T, simd_type, simd_flags>(p);
    }

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

} // namespace impl
} // namespace bla
