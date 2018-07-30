#pragma once

#include "allscale/api/user/data/impl/expressions/expression.h"
#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/types.h"

#include <Vc/Vc>

namespace allscale {
namespace api {
namespace user {
namespace data {
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

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return -expression.template packet<simd_type>(p);
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
