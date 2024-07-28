#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/traits.h"
#include "bla/impl/types.h"

#include <Vc/Vc>

namespace bla {
namespace impl {

template <typename E1, typename E2>
class MatrixSubtraction : public MatrixExpression<MatrixSubtraction<E1, E2>> {
    static_assert(is_valid_v<std::minus<>, scalar_type_t<E1>, scalar_type_t<E2>>, "No - implementation for these MatrixExpression types.");

    using typename MatrixExpression<MatrixSubtraction<E1, E2>>::T;
    using typename MatrixExpression<MatrixSubtraction<E1, E2>>::PacketScalar;

    using Exp1 = expression_member_t<E1>;
    using Exp2 = expression_member_t<E2>;

public:
    MatrixSubtraction(Exp1 u, Exp2 v) : lhs(u), rhs(v) {
        assert_eq(lhs.size(), rhs.size());
    }

    T operator[](const point_type& pos) const {
        return lhs[pos] - rhs[pos];
    }

    point_type size() const {
        return rhs.size();
    }

    coordinate_type rows() const {
        return lhs.rows();
    }

    coordinate_type columns() const {
        return lhs.columns();
    }

    template <typename T, typename simd_type = PacketScalar, typename simd_flags = Vc::UnalignedTag>
    simd_type packet(T p) const {
        return lhs.template packet<T, simd_type, simd_flags>(p) - rhs.template packet<T, simd_type, simd_flags>(p);
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
} // namespace bla
