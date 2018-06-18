#pragma once

#include "allscale/api/user/data/impl/traits.h"
#include "allscale/api/user/data/impl/types.h"

#include <Eigen/Dense>
#include <Vc/Vc>
#include <algorithm>
#include <allscale/api/user/data/grid.h>
#include <allscale/utils/assert.h>
#include <allscale/utils/vector.h>
#include <array>
#include <cmath>
#include <complex>
#include <functional>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

namespace detail {

template <typename T1, typename T2>
std::enable_if_t<vectorizable_v<Matrix<T2>>> set_value(const T1& value, Matrix<T2>& dst) {
    using PacketScalar = typename Vc::native_simd<T2>;


    const int total_size = dst.rows() * dst.columns();
    const int packet_size = PacketScalar::size();
    const int aligned_end = total_size / packet_size * packet_size;

    algorithm::pfor(utils::Vector<coordinate_type, 1>(0), utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
        int i = coord[0] * packet_size;
        point_type p{i / dst.columns(), i % dst.columns()};

        PacketScalar z(static_cast<T2>(value));

        z.copy_to(&dst[p], alignment_t<PacketScalar>{});
    });

    for(int i = aligned_end; i < total_size; i++) {
        point_type p{i / dst.columns(), i % dst.columns()};
        dst[p] = static_cast<T2>(value);
    }
}

template <typename T1, typename T2>
std::enable_if_t<!vectorizable_v<Matrix<T2>>> set_value(const T1& value, Matrix<T2>& dst) {
    algorithm::pfor(dst.size(), [&](const auto& pos) { dst[pos] = static_cast<T2>(value); });
}

template <typename T1, typename T2>
void set_value(const T1& value, RefSubMatrix<T2, true>& dst) {
    using PacketScalar = typename Vc::native_simd<T2>;


    const int total_size = dst.rows() * dst.columns();
    const int packet_size = PacketScalar::size();
    const int aligned_end = total_size / packet_size * packet_size;

    algorithm::pfor(utils::Vector<coordinate_type, 1>(0), utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
        int i = coord[0] * packet_size;
        point_type p{i / dst.columns(), i % dst.columns()};

        PacketScalar z(static_cast<T2>(value));

        z.copy_to(std::addressof(dst[p]), alignment_t<PacketScalar>{});
    });

    for(int i = aligned_end; i < total_size; i++) {
        point_type p{i / dst.columns(), i % dst.columns()};
        dst[p] = static_cast<T2>(value);
    }
}

template <typename T1, typename T2>
void set_value(const T1& value, RefSubMatrix<T2, false>& dst) {
    algorithm::pfor(dst.size(), [&](const auto& pos) { dst[pos] = static_cast<T2>(value); });
}


template <typename T>
T conj(const T& x) {
    return x;
}

template <typename T>
std::complex<T> conj(const std::complex<T>& x) {
    return std::conj(x);
}

template <bool Contiguous = false, typename E>
auto sub(const MatrixExpression<E>& e, const BlockRange& br) {
    return SubMatrix<E>(e, br);
}

template <bool Contiguous = false, typename T>
auto sub(Matrix<T>& e, const BlockRange& br) {
    return RefSubMatrix<T, Contiguous>(e, br);
}

template <bool Contiguous = false, typename T>
auto sub(const RefSubMatrix<T>& e, BlockRange block_range) {
    assert_ge(block_range.start, (point_type{0, 0}));
    assert_le(block_range.start + block_range.size, e.getBlockRange().size);

    BlockRange new_range;
    new_range.start = e.getBlockRange().start + block_range.start;
    new_range.size = block_range.size;
    return RefSubMatrix<T, Contiguous>(e.getExpression(), new_range);
}

template <bool Contiguous = false, typename E>
auto sub(const SubMatrix<E>& e, BlockRange block_range) {
    assert_ge(block_range.start, (point_type{0, 0}));
    assert_le(block_range.start + block_range.size, e.getBlockRange().size);

    BlockRange new_range;
    new_range.start = e.getBlockRange().start + block_range.start;
    new_range.size = block_range.size;
    return SubMatrix<E>(e.getExpression(), new_range);
}

template <typename E>
auto row(const MatrixExpression<E>& e, coordinate_type r) {
    assert_lt(r, e.rows());
    return sub(e, {{r, 0}, {1, e.columns()}});
}

template <typename T>
auto row(Matrix<T>& e, coordinate_type r) {
    assert_lt(r, e.rows());
    return sub(e, {{r, 0}, {1, e.columns()}});
}

template <typename T, bool C>
auto row(const RefSubMatrix<T, C>& e, coordinate_type r) {
    assert_lt(r, e.rows());
    return sub(e, {{r, 0}, {1, e.columns()}});
}

template <typename E>
auto column(const MatrixExpression<E>& e, coordinate_type c) {
    assert_lt(c, e.columns());
    return sub(e, {{0, c}, {e.rows(), 1}});
}

template <typename T>
auto column(Matrix<T>& e, coordinate_type c) {
    assert_lt(c, e.columns());
    return sub(e, {{0, c}, {e.rows(), 1}});
}

template <typename T, bool C>
auto column(const RefSubMatrix<T, C>& e, coordinate_type c) {
    assert_lt(c, e.columns());
    return sub(e, {{0, c}, {e.rows(), 1}});
}


} // end namespace detail

template <typename E>
class MatrixExpression {
    static_assert(std::is_same<E, detail::remove_cvref_t<E>>::value, "A MatrixExpression type may not be cv qualified.");


public:
    using T = scalar_type_t<E>;
    using PacketScalar = typename Vc::native_simd<T>;

    /*
     * abstract class due to object slicing
     */
protected:
    MatrixExpression() = default;
    MatrixExpression(const MatrixExpression&) = default;
    MatrixExpression& operator=(const MatrixExpression&) = delete;

public:
    T operator[](const point_type& pos) const {
        assert_lt(pos, size());
        assert_ge(pos, (point_type{0, 0}));
        return impl()[pos];
    }

    point_type size() const {
        return impl().size();
    }

    coordinate_type rows() const {
        return impl().rows();
    }

    coordinate_type columns() const {
        return impl().columns();
    }

    bool isSquare() const {
        return rows() == columns();
    }

    auto row(coordinate_type r) {
        return detail::row(impl(), r);
    }

    auto row(coordinate_type r) const {
        return detail::row(impl(), r);
    }

    auto column(coordinate_type c) {
        return detail::column(impl(), c);
    }

    auto column(coordinate_type c) const {
        return detail::column(impl(), c);
    }

    auto rowRange(range_type p) {
        return detail::sub<false>(impl(), {{p.x, 0}, {p.y, columns()}});
    }

    auto rowRange(range_type p) const {
        return detail::sub<false>(impl(), {{p.x, 0}, {p.y, columns()}});
    }

    auto columnRange(range_type p) {
        return detail::sub<false>(impl(), {{0, p.x}, {rows(), p.y}});
    }

    auto columnRange(range_type p) const {
        return detail::sub<false>(impl(), {{0, p.x}, {rows(), p.y}});
    }

    auto topRows(coordinate_type row_count) {
        return rowRange({0, row_count});
    }

    auto topRows(coordinate_type row_count) const {
        return rowRange({0, row_count});
    }

    auto bottomRows(coordinate_type row_count) {
        return rowRange({rows() - row_count, row_count});
    }

    auto bottomRows(coordinate_type row_count) const {
        return rowRange({rows() - row_count, row_count});
    }

    auto topColumns(coordinate_type column_count) {
        return columnRange({0, column_count});
    }

    auto topColumns(coordinate_type column_count) const {
        return columnRange({0, column_count});
    }

    auto bottomColumns(coordinate_type column_count) {
        return columnRange({columns() - column_count, column_count});
    }

    auto bottomColumns(coordinate_type column_count) const {
        return columnRange({columns() - column_count, column_count});
    }

    template <typename E2>
    ElementMatrixMultiplication<E, E2> product(const MatrixExpression<E2>& e) const {
        return ElementMatrixMultiplication<E, E2>(impl(), e);
    }

    MatrixTranspose<E> transpose() const;

    MatrixConjugate<E> conjugate() const {
        return MatrixConjugate<E>(impl());
    }

    MatrixTranspose<MatrixConjugate<E>> adjoint() const {
        return this->conjugate().transpose();
    }

    auto sub(BlockRange block_range) {
        return detail::sub(impl(), block_range);
    }

    auto sub(BlockRange block_range) const {
        return detail::sub(impl(), block_range);
    }

    MatrixAbs<E> abs() const {
        return MatrixAbs<E>(impl());
    }

    T norm() const {
        return std::sqrt(product(*this).reduce(0, std::plus<T>{}));
    }

    // -- defined in decomposition.h
    LUD<T> LUDecomposition() const;
    QRD<T> QRDecomposition() const;
    SVD<T> SVDecomposition() const;

    template <typename Reducer>
    T reduce(T init, Reducer f) const {
        using ct = coordinate_type;
        T result = init;
        // TODO: use preduce
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                result = f(result, (*this)[{i, j}]);
            }
        }

        return result;
    }

    template <typename Reducer>
    T reduce(Reducer f) const {
        using ct = coordinate_type;

        T result{};
        bool first = true;

        // TODO: use preduce
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                if(first) {
                    first = false;
                    result = (*this)[{i, j}];
                    continue;
                }
                result = f(result, (*this)[{i, j}]);
            }
        }

        return result;
    }

    T max() const {
        return reduce([](T a, T b) { return std::max(a, b); });
    }

    T min() const {
        return reduce([](T a, T b) { return std::min(a, b); });
    }

    // -- defined in decomposition.h
    T determinant() const;
    Matrix<T> inverse() const;

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    std::enable_if_t<vectorizable_v<E>, simd_type> packet(point_type p) const {
        return impl().template packet<simd_type, align>(p);
    }

    // -- defined in evaluate.h
    auto eval();
    auto eval() const;

    operator E&() {
        return impl();
    }

    operator const E&() const {
        return impl();
    }

private:
    E& impl() {
        return static_cast<E&>(*this);
    }

    const E& impl() const {
        return static_cast<const E&>(*this);
    }
};

template <typename E1, typename E2>
class MatrixAddition : public MatrixExpression<MatrixAddition<E1, E2>> {
    static_assert(is_valid_v<std::plus<>, scalar_type_t<E1>, scalar_type_t<E2>>, "No + implementation for these MatrixExpression types.");

    using typename MatrixExpression<MatrixAddition<E1, E2>>::T;
    using typename MatrixExpression<MatrixAddition<E1, E2>>::PacketScalar;

    using Exp1 = expression_member_t<E1>;
    using Exp2 = expression_member_t<E2>;

public:
    MatrixAddition(Exp1 u, Exp2 v) : lhs(u), rhs(v) {
        assert(u.size() == v.size());
    }

    T operator[](const point_type& pos) const {
        return lhs[pos] + rhs[pos];
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

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return lhs.template packet<simd_type, align>(p) + rhs.template packet<simd_type, align>(p);
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

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return lhs.template packet<simd_type, align>(p) - rhs.template packet<simd_type, align>(p);
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

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return lhs.template packet<simd_type, align>(p) * rhs.template packet<simd_type, align>(p);
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

template <typename E>
class MatrixTranspose : public MatrixExpression<MatrixTranspose<E>> {
    using typename MatrixExpression<MatrixTranspose<E>>::T;

    using Exp = expression_member_t<E>;

public:
    MatrixTranspose(Exp u) : expression(u) {
    }

    T operator[](const point_type& pos) const {
        return expression[{pos.y, pos.x}];
    }

    void evaluation(Matrix<T>& tmp) {
        using ct = coordinate_type;

        using block_type = SimdBlock<decltype(expression.packet({0, 0}))>;

        algorithm::pfor(point_type{rows() / block_type::size()[0], columns() / block_type::size()[1]}, [&](const auto& pos) {
            coordinate_type i = pos.x * block_type::size()[0];
            coordinate_type j = pos.y * block_type::size()[1];
            block_type b(expression, {j, i});

            b.transpose();
            b.load_to(tmp, {i, j});
        });


        // transpose the rest that can't be done with a full block
        // right side
        for(ct i = 0; i < rows() - rows() % block_type::size()[0]; ++i) {
            for(ct j = columns() - columns() % block_type::size()[1]; j < columns(); ++j) {
                tmp[{i, j}] = expression[{j, i}];
            }
        }

        // bottom
        for(ct i = rows() - rows() % block_type::size()[0]; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                tmp[{i, j}] = expression[{j, i}];
            }
        }
    }

    point_type size() const {
        return {rows(), columns()};
    }

    coordinate_type rows() const {
        return expression.columns();
    }

    coordinate_type columns() const {
        return expression.rows();
    }

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

template <typename E>
class MatrixConjugate : public MatrixExpression<MatrixConjugate<E>> {
    using typename MatrixExpression<MatrixConjugate<E>>::T;

    using Exp = expression_member_t<E>;

public:
    MatrixConjugate(Exp u) : expression(u) {
    }

    T operator[](const point_type& pos) const {
        return detail::conj(expression[pos]);
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

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

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

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return -expression.template packet<simd_type, align>(p);
    }

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

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

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return Vc::abs(expression.template packet<simd_type, align>(p));
    }

    Exp getExpression() const {
        return expression;
    }

private:
    Exp expression;
};

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

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return expression.template packet<simd_type, align>(p) * simd_type(scalar);
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

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return simd_type(scalar) * expression.template packet<simd_type, align>(p);
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

template <typename T>
class Matrix : public MatrixExpression<Matrix<T>> {
    using map_type = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using cmap_type = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

    using typename MatrixExpression<Matrix<T>>::PacketScalar;

public:
    Matrix(const point_type& size) : m_data(size) {
    }

    template <typename E>
    Matrix(const MatrixExpression<E>& mat) : m_data(mat.size()) {
        evaluate(mat);
    }

    template <typename Derived>
    Matrix(const Eigen::MatrixBase<Derived>& matrix) : m_data({matrix.rows(), matrix.cols()}) {
        algorithm::pfor(size(), [&](const point_type& p) { m_data[p] = matrix(p.x, p.y); });
    }

    Matrix(const Matrix& mat) : MatrixExpression<Matrix<T>>(), m_data(mat.size()) {
        evaluate(mat);
    }

    Matrix(Matrix&&) = default;

    Matrix& operator=(const Matrix& mat) {
        evaluate(mat);

        return *this;
    }

    Matrix& operator=(Matrix&&) = default;

    Matrix& operator=(const T& value) {
        fill(value);
        return (*this);
    }

    template <typename E>
    Matrix& operator=(MatrixExpression<E> const& mat) {
        evaluate(mat);

        return *this;
    }

    T& operator[](const point_type& pos) {
        return m_data[pos];
    }

    const T& operator[](const point_type& pos) const {
        return m_data[pos];
    }

    point_type size() const {
        return m_data.size();
    }

    coordinate_type rows() const {
        return m_data.size()[0];
    }

    coordinate_type columns() const {
        return m_data.size()[1];
    }

    map_type eigenSub(const range_type& r) {
        return map_type(&m_data[{r.x, 0}], r.y, columns());
    }

    cmap_type eigenSub(const range_type& r) const {
        return cmap_type(&m_data[{r.x, 0}], r.y, columns());
    }

    void fill(const T& value) {
        detail::set_value(value, *this);
    }

    void fill(std::function<T(point_type)> f) {
        algorithm::pfor(m_data.size(), [&](const point_type& p) { m_data[p] = f(p); });
    }

    void fill(std::function<T()> f) {
        algorithm::pfor(m_data.size(), [&](const point_type& p) { m_data[p] = f(); });
    }

    void fill_seq(const T& value) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = value;
            }
        }
    }

    void fill_seq(std::function<T(point_type)> f) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = f(point_type{i, j});
            }
        }
    }

    void fill_seq(std::function<T()> f) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = f();
            }
        }
    }

    void zero() {
        fill(static_cast<T>(0));
    }

    void eye() {
        fill([](const auto& pos) { return pos.x == pos.y ? static_cast<T>(1) : static_cast<T>(0); });
    }

    void identity() {
        assert_eq(rows(), columns());
        eye();
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> toEigenMatrix() {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(rows(), columns());
        algorithm::pfor(size(), [&](const point_type& p) { result(p.x, p.y) = m_data[p]; });
        return result;
    }

    map_type getEigenMap() {
        return eigenSub({0, rows()});
    }

    cmap_type getEigenMap() const {
        return eigenSub({0, rows()});
    }

    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p), align{});
    }

    const Matrix<T>& eval() const {
        return *this;
    }
    Matrix<T>& eval() {
        return *this;
    }

private:
    template <typename E>
    void evaluate(const MatrixExpression<E>&);

    data::Grid<T, 2> m_data;
};

template <typename T>
class PermutationMatrix : public MatrixExpression<PermutationMatrix<T>> {
public:
    PermutationMatrix(coordinate_type c) : values(utils::Vector<coordinate_type, 1>{c}), swaps(0) {
        algorithm::pfor(utils::Vector<coordinate_type, 1>{c}, [&](const auto& p) { values[p] = p[0]; });
    }

    // TODO: remove this
    PermutationMatrix(const PermutationMatrix<T>& mat)
        : MatrixExpression<PermutationMatrix<T>>(), values(utils::Vector<coordinate_type, 1>{mat.rows()}), swaps(0) {
        algorithm::pfor(utils::Vector<coordinate_type, 1>{rows()}, [&](const auto& p) { values[p] = mat.values[p]; });
    }

    PermutationMatrix(PermutationMatrix<T>&&) = default;

    T operator[](const point_type& pos) const {
        assert_lt(pos, size());
        return values[{pos.x}] == pos.y ? static_cast<T>(1) : static_cast<T>(0);
    }

    point_type size() const {
        return {rows(), columns()};
    }

    coordinate_type rows() const {
        return values.size()[0];
    }

    coordinate_type columns() const {
        return values.size()[0];
    }

    void swap(coordinate_type i, coordinate_type j) {
        if(i == j)
            return;

        coordinate_type old = values[{i}];
        values[{i}] = values[{j}];
        values[{j}] = old;
        swaps++;
    }

    coordinate_type permutation(coordinate_type i) const {
        assert_lt(i, rows());
        return values[i];
    }

    int numSwaps() const {
        return swaps;
    }

private:
    Grid<coordinate_type, 1> values;
    int swaps;
};

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

private:
    Exp expression;
    BlockRange block_range;
};

template <typename T, bool Contiguous>
class RefSubMatrix : public MatrixExpression<RefSubMatrix<T, Contiguous>> {
    using typename MatrixExpression<RefSubMatrix<T, Contiguous>>::PacketScalar;

    using Exp = detail::remove_cvref_t<Matrix<T>>;

public:
    RefSubMatrix(Exp& m) : expression(m), block_range({{0, 0}, {m.size()}}) {
    }

    RefSubMatrix(Exp& v, BlockRange block_range) : expression(v), block_range(block_range) {
        assert_ge(block_range.start, (point_type{0, 0}));
        assert_ge(block_range.size, (point_type{0, 0}));
        assert_le(block_range.start + block_range.size, expression.size());
    }

    template <typename E>
    RefSubMatrix& operator=(const MatrixExpression<E>& exp) {
        assert_eq(size(), exp.size());
        algorithm::pfor(size(), [&](const point_type& p) { (*this)[p] = exp[p]; });

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

    void fill(const T& value) {
        detail::set_value(value, *this);
    }

    void fill(std::function<T(point_type)> f) {
        algorithm::pfor(size(), [&](const point_type& p) { (*this)[p] = f(p); });
    }

    void fill(std::function<T()> f) {
        algorithm::pfor(size(), [&](const point_type& p) { (*this)[p] = f(); });
    }

    void fill_seq(const T& value) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = value;
            }
        }
    }

    void fill_seq(std::function<T(point_type)> f) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = f(point_type{i, j});
            }
        }
    }

    void fill_seq(std::function<T()> f) {
        using ct = coordinate_type;
        for(ct i = 0; i < rows(); ++i) {
            for(ct j = 0; j < columns(); ++j) {
                (*this)[{i, j}] = f();
            }
        }
    }

    void zero() {
        fill(static_cast<T>(0));
    }

    void eye() {
        fill([](const auto& pos) { return pos.x == pos.y ? static_cast<T>(1) : static_cast<T>(0); });
    }

    void identity() {
        assert_eq(rows(), columns());
        eye();
    }

    template <typename E2, bool C2>
    void swap(RefSubMatrix<E2, C2> other) {
        assert_eq(size(), other.size());

        algorithm::pfor(size(), [&](const auto& pos) {
            T tmp = (*this)[pos];
            (*this)[pos] = other[pos];
            other[pos] = tmp;
        });
    }
    template <typename simd_type = PacketScalar, typename align = Vc::flags::element_aligned_tag>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p), align{});
    }

    Exp& getExpression() const {
        return expression;
    }

    BlockRange getBlockRange() const {
        return block_range;
    }

private:
    Exp& expression;
    BlockRange block_range;
};

template <typename T>
class IdentityMatrix : public MatrixExpression<IdentityMatrix<T>> {
public:
    IdentityMatrix(point_type matrix_size) : matrix_size(matrix_size) {
    }

    T operator[](const point_type& pos) const {
        assert_lt(pos, matrix_size);
        return pos.x == pos.y ? static_cast<T>(1) : static_cast<T>(0);
    }

    point_type size() const {
        return matrix_size;
    }

    coordinate_type rows() const {
        return matrix_size[0];
    }

    coordinate_type columns() const {
        return matrix_size[1];
    }

private:
    point_type matrix_size;
};

} // end namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
