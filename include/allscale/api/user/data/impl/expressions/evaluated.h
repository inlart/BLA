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

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return tmp.template packet<simd_type, align>(p);
    }

    //    EvaluatedExpression(const EvaluatedExpression&) = delete;
    //    EvaluatedExpression(EvaluatedExpression&&) = default;
    //
    //    EvaluatedExpression& operator=(const EvaluatedExpression&) = delete;
    //    EvaluatedExpression& operator=(EvaluatedExpression&&) = default;

private:
    Matrix<T> tmp;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
