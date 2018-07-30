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

template <typename E, bool V>
class SubMatrix : public MatrixExpression<SubMatrix<E, V>> {
    using typename MatrixExpression<SubMatrix<E, V>>::T;

    using Exp = expression_member_t<E>;

public:
    static constexpr bool is_vector = V;

    SubMatrix(Exp v, BlockRange block_range) : expression(v), block_range(block_range) {
        assert_ge(block_range.start, (point_type{0, 0}));
        assert_ge(block_range.size, (point_type{0, 0}));
        assert_le(block_range.start + block_range.size, expression.size());
    }

    template <bool V2>
    SubMatrix(SubMatrix<E, V2> sub) : expression(sub.getExpression()), block_range(sub.getBlockRange()) {
    }

    T operator[](const point_type& pos) const {
        return expression[pos + block_range.start];
    }

    point_type size() const {
        return block_range.size;
    }
    coordinate_type rows() const {
        return block_range.size[0];
    }

    coordinate_type columns() const {
        return block_range.size[1];
    }

    Exp getExpression() const {
        return expression;
    }

    BlockRange getBlockRange() const {
        return block_range;
    }

    coordinate_type stride() const {
        return expression.stride();
    }

    // TODO: packet

private:
    Exp expression;
    BlockRange block_range;
};

// TODO: is there a better way to do this?
template <typename T, bool V>
class SubMatrix<const Matrix<T>, V> : public MatrixExpression<SubMatrix<const Matrix<T>, V>> {
    using typename MatrixExpression<SubMatrix<const Matrix<T>, V>>::PacketScalar;
    using Exp = expression_member_t<const Matrix<T>>;

public:
    static constexpr bool is_vector = V;

    SubMatrix(Exp m) : expression(m), block_range({{0, 0}, {m.size()}}) {
    }

    SubMatrix(Exp v, BlockRange block_range) : expression(v), block_range(block_range) {
        assert_ge(block_range.start, (point_type{0, 0}));
        assert_ge(block_range.size, (point_type{0, 0}));
        assert_le(block_range.start + block_range.size, expression.size());
    }

    template <bool V2>
    SubMatrix(SubMatrix<const Matrix<T>, V2> sub) : expression(sub.getExpression()), block_range(sub.getBlockRange()) {
    }

    const T& operator[](const point_type& pos) const {
        return expression[pos + block_range.start];
    }

    point_type size() const {
        return block_range.size;
    }
    coordinate_type rows() const {
        return block_range.size[0];
    }

    coordinate_type columns() const {
        return block_range.size[1];
    }

    Exp getExpression() const {
        return expression;
    }

    BlockRange getBlockRange() const {
        return block_range;
    }

    coordinate_type stride() const {
        return expression.stride();
    }

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p));
    }

private:
    Exp expression;
    BlockRange block_range;
};

template <typename T, bool V>
class SubMatrix<Matrix<T>, V> : public AccessBase<SubMatrix<Matrix<T>, V>> {
    using typename MatrixExpression<SubMatrix<Matrix<T>, V>>::PacketScalar;
    using Exp = expression_member_t<Matrix<T>>;

public:
    static constexpr bool is_vector = V;

    SubMatrix(Exp& m) : expression(m), block_range({{0, 0}, {m.size()}}) {
    }

    SubMatrix(Exp& v, BlockRange block_range) : expression(v), block_range(block_range) {
        assert_ge(block_range.start, (point_type{0, 0}));
        assert_ge(block_range.size, (point_type{0, 0}));
        assert_le(block_range.start + block_range.size, expression.size());
    }

    template <bool V2>
    SubMatrix(SubMatrix<Matrix<T>, V2> sub) : expression(sub.getExpression()), block_range(sub.getBlockRange()) {
    }

    template <typename E2>
    SubMatrix<Matrix<T>, V>& operator=(const MatrixExpression<E2>& mat) {
        AccessBase<SubMatrix<Matrix<T>>>::evaluate(mat);

        return *this;
    }

    const T& operator[](const point_type& pos) const {
        return expression[pos + block_range.start];
    }

    T& operator[](const point_type& pos) {
        return expression[pos + block_range.start];
    }

    point_type size() const {
        return block_range.size;
    }
    coordinate_type rows() const {
        return block_range.size[0];
    }

    coordinate_type columns() const {
        return block_range.size[1];
    }

    // -- defined in evaluate.h
    template <typename T2, bool V2>
    void swap(SubMatrix<Matrix<T2>, V2> other);

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p));
    }

    coordinate_type stride() const {
        return expression.stride();
    }

    Exp& getExpression() const {
        return expression;
    }

    BlockRange getBlockRange() const {
        return block_range;
    }

private:
    Exp expression;
    BlockRange block_range;
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
