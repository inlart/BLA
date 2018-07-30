#pragma once

#include "allscale/api/user/data/impl/expressions/expression.h"
#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/traits.h"
#include "allscale/api/user/data/impl/types.h"

#include <Vc/Vc>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename E1, typename E2>
class ElementMatrixMultiplication : public MatrixExpression<ElementMatrixMultiplication<E1, E2>> {
    static_assert(is_valid_v<std::multiplies<>, scalar_type_t<E1>, scalar_type_t<E2>>, "No * implementation for these MatrixExpression types.");

    using typename MatrixExpression<ElementMatrixMultiplication<E1, E2>>::T;
    using typename MatrixExpression<ElementMatrixMultiplication<E1, E2>>::PacketScalar;

    using Exp1 = expression_member_t<E1>;
    using Exp2 = expression_member_t<E2>;

public:
    ElementMatrixMultiplication(Exp1 u, Exp2 v) : lhs(u), rhs(v) {
        assert_eq(lhs.size(), rhs.size());
    }

    T operator[](const point_type& pos) const {
        return lhs[pos] * rhs[pos];
    }

    point_type size() const {
        return lhs.size();
    }

    coordinate_type rows() const {
        return lhs.rows();
    }

    coordinate_type columns() const {
        return lhs.columns();
    }

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return lhs.template packet<simd_type>(p) * rhs.template packet<simd_type>(p);
    }

    Exp1 getLeftExpression() const {
        return lhs;
    }

    Exp2 getRightExpression() const {
        return rhs;
    }

private:
    Exp1 lhs;
    Exp2 rhs;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
