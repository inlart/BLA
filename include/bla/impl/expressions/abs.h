#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/traits.h"
#include "bla/impl/types.h"

#include <Vc/Vc>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename E>
class MatrixAbs : public MatrixExpression<MatrixAbs<E>> {
    using typename MatrixExpression<MatrixAbs<E>>::T;
    using typename MatrixExpression<MatrixAbs<E>>::PacketScalar;

    using Exp = expression_member_t<E>;

public:
    MatrixAbs(Exp e) : expression(e) {
    }
    T operator[](const point_type& pos) const {
        return std::abs(expression[pos]);
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

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return Vc::abs(expression.template packet<simd_type>(p));
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
