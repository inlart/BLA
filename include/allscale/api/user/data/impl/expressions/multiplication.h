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

template <typename E1, typename E2>
class MatrixMultiplication : public MatrixExpression<MatrixMultiplication<E1, E2>> {
    static_assert(is_valid_v<std::multiplies<>, scalar_type_t<E1>, scalar_type_t<E2>>, "No * implementation for these MatrixExpression types.");

    using intermediate_type = operation_result_t<std::multiplies<>, scalar_type_t<E1>, scalar_type_t<E2>>;

    static_assert(is_valid_v<std::plus<>, intermediate_type, intermediate_type> && type_consistent_v<std::plus<>, intermediate_type>,
                  "No valid + implementation for these MatrixExpression types.");

    using typename MatrixExpression<MatrixMultiplication<E1, E2>>::T;
    using typename MatrixExpression<MatrixMultiplication<E1, E2>>::PacketScalar;

    using Exp1 = expression_member_t<E1>;
    using Exp2 = expression_member_t<E2>;

public:
    MatrixMultiplication(Exp1 u, Exp2 v) : lhs(u), rhs(v) {
        assert_eq(lhs.columns(), rhs.rows());
    }

    T operator[](const point_type& pos) const {
        T value = static_cast<T>(0);

        // TODO: preduce?
        for(int i = 0; i < lhs.columns(); ++i) {
            value += lhs[{pos.x, i}] * rhs[{i, pos.y}];
        }

        return value;
    }

    point_type size() const {
        return {rows(), columns()};
    }

    coordinate_type rows() const {
        return lhs.rows();
    }

    coordinate_type columns() const {
        return rhs.columns();
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
