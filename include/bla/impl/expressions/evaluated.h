#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/types.h"

#include <Vc/Vc>

namespace bla {
namespace impl {

// -- A wrapper around a temporary Matrix
template <typename T>
class EvaluatedExpression : public MatrixExpression<EvaluatedExpression<T>> {
    using typename MatrixExpression<EvaluatedExpression<T>>::PacketScalar;

public:
    EvaluatedExpression(Matrix<T>&& m) : tmp(std::move(m)) {
    }

    T operator[](const point_type& pos) const {
        return tmp[pos];
    }

    point_type size() const {
        return {rows(), columns()};
    }

    coordinate_type rows() const {
        return tmp.rows();
    }

    coordinate_type columns() const {
        return tmp.columns();
    }

    template <typename S, typename simd_type = PacketScalar, typename simd_flags = Vc::UnalignedTag>
    simd_type packet(S p) const {
        return tmp.template packet<S, simd_type, simd_flags>(p);
    }

    Matrix<T>&& getMatrix() && {
        return std::move(tmp);
    }

private:
    Matrix<T> tmp;
};

} // namespace impl
} // namespace bla
