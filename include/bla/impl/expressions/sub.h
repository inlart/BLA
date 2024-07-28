#pragma once

#include "bla/impl/expressions/expression.h"
#include "bla/impl/forward.h"
#include "bla/impl/traits.h"
#include "bla/impl/types.h"

#include <Vc/Vc>

namespace bla {
namespace impl {

template <typename E>
class SubMatrix : public MatrixExpression<SubMatrix<E>> {
    using typename MatrixExpression<SubMatrix<E>>::T;

    using Exp = expression_member_t<E>;

public:
    SubMatrix(Exp v, BlockRange block_range) : expression(v), block_range(block_range) {
        assert_ge(block_range.start, (point_type{0, 0}));
        assert_ge(block_range.size, (point_type{0, 0}));
        assert_le(block_range.start + block_range.size, expression.size());
    }

    SubMatrix(const SubMatrix<E>& sub) : expression(sub.getExpression()), block_range(sub.getBlockRange()) {
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
template <typename T>
class SubMatrix<const Matrix<T>> : public MatrixExpression<SubMatrix<const Matrix<T>>> {
    using typename MatrixExpression<SubMatrix<const Matrix<T>>>::PacketScalar;
    using Exp = expression_member_t<const Matrix<T>>;

public:
    SubMatrix(Exp m) : expression(m), block_range({{0, 0}, {m.size()}}) {
    }

    SubMatrix(Exp v, BlockRange block_range) : expression(v), block_range(block_range) {
        assert_ge(block_range.start, (point_type{0, 0}));
        assert_ge(block_range.size, (point_type{0, 0}));
        assert_le(block_range.start + block_range.size, expression.size());
    }

    SubMatrix(const SubMatrix<const Matrix<T>>& sub) : expression(sub.getExpression()), block_range(sub.getBlockRange()) {
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

    template <typename S, typename simd_type = PacketScalar, typename simd_flags = Vc::UnalignedTag>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p), simd_flags{});
    }

private:
    Exp expression;
    BlockRange block_range;
};

template <typename T>
class SubMatrix<Matrix<T>> : public AccessBase<SubMatrix<Matrix<T>>> {
    using typename MatrixExpression<SubMatrix<Matrix<T>>>::PacketScalar;
    using Exp = expression_member_t<Matrix<T>>;

public:
    SubMatrix(Exp& m) : expression(m), block_range({{0, 0}, {m.size()}}) {
    }

    SubMatrix(Exp& v, BlockRange block_range) : expression(v), block_range(block_range) {
        assert_ge(block_range.start, (point_type{0, 0}));
        assert_ge(block_range.size, (point_type{0, 0}));
        assert_le(block_range.start + block_range.size, expression.size());
    }

    SubMatrix(const SubMatrix<Matrix<T>>& sub) : expression(sub.getExpression()), block_range(sub.getBlockRange()) {
    }

    SubMatrix<Matrix<T>>& operator=(const SubMatrix<Matrix<T>>& mat) {
        AccessBase<SubMatrix<Matrix<T>>>::evaluate(mat);

        return *this;
    }

    template <typename E2>
    SubMatrix<Matrix<T>>& operator=(const MatrixExpression<E2>& mat) {
        AccessBase<SubMatrix<Matrix<T>>>::evaluate(mat);

        return *this;
    }

    const T& operator[](const point_type& pos) const {
        return expression[pos + block_range.start];
    }

    T& operator[](const point_type& pos) {
        return expression[pos + block_range.start];
    }

    operator SubMatrix<const Matrix<T>>() const {
        return SubMatrix<const Matrix<T>>(expression, block_range);
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
    template <typename T2>
    void swap(SubMatrix<Matrix<T2>> other);

    template <typename S, typename simd_type = PacketScalar, typename simd_flags = Vc::UnalignedTag>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p), simd_flags{});
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
} // namespace bla
