#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/traits.h"
#include "bla/impl/types.h"

#include <Vc/Vc>

namespace bla {
namespace impl {

template <typename E, typename U>
class MatrixScalarMultiplication : public MatrixExpression<MatrixScalarMultiplication<E, U>> {
    using typename MatrixExpression<MatrixScalarMultiplication<E, U>>::T;
    using typename MatrixExpression<MatrixScalarMultiplication<E, U>>::PacketScalar;

    using Exp = expression_member_t<E>;

public:
    MatrixScalarMultiplication(Exp v, const U& u) : scalar(u), expression(v) {
    }

    T operator[](const point_type& pos) const {
        return expression[pos] * scalar;
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

    template <typename S, typename simd_type = PacketScalar, typename simd_flags = Vc::UnalignedTag>
    simd_type packet(S p) const {
        return expression.template packet<S, simd_type, simd_flags>(p) * simd_type(scalar);
    }

    const U& getScalar() const {
        return scalar;
    }

    Exp getExpression() const {
        return expression;
    }

private:
    const U scalar;
    Exp expression;
};

template <typename E, typename U>
class ScalarMatrixMultiplication : public MatrixExpression<ScalarMatrixMultiplication<E, U>> {
    using typename MatrixExpression<ScalarMatrixMultiplication<E, U>>::T;
    using typename MatrixExpression<ScalarMatrixMultiplication<E, U>>::PacketScalar;

    using Exp = expression_member_t<E>;

public:
    ScalarMatrixMultiplication(const U& u, Exp v) : scalar(u), expression(v) {
    }

    T operator[](const point_type& pos) const {
        return scalar * expression[pos];
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
        return simd_type(scalar) * expression.template packet<T, simd_type, simd_flags>(p);
    }
    const U& getScalar() const {
        return scalar;
    }

    Exp getExpression() const {
        return expression;
    }

private:
    const U scalar;
    Exp expression;
};

} // namespace impl
} // namespace bla
