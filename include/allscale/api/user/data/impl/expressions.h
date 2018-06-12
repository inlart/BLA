#pragma once

#include "allscale/api/user/data/impl/decomposition.h"
#include "allscale/api/user/data/impl/traits.h"
//#include "allscale/api/user/data/impl/transpose.h"
#include "allscale/api/user/data/impl/types.h"

#include "allscale/api/user/data/impl/forward.h"

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

#ifdef Vc_HAVE_SSE
#include <xmmintrin.h>
#endif

#ifdef Vc_HAVE_AVX
#include <immintrin.h>
#endif

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

        z.copy_to(&dst[p], Vc::flags::element_aligned);
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

        z.copy_to(std::addressof(dst[p]), Vc::flags::element_aligned);
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

// -- evaluate a matrix expression using vectorization
template <typename E>
std::enable_if_t<vectorizable_v<E>> evaluate(const MatrixExpression<E>& expression, scalar_type_t<E>* dst) {
    expression_member_t<decltype(simplify(expression))> expr = simplify(expression);

    using T = scalar_type_t<E>;
    using PacketScalar = typename Vc::native_simd<T>;


    const int total_size = expr.rows() * expr.columns();
    const int packet_size = PacketScalar::size();
    const int aligned_end = total_size / packet_size * packet_size;

    algorithm::pfor(utils::Vector<coordinate_type, 1>(0), utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
        int i = coord[0] * packet_size;
        point_type p{i / expr.columns(), i % expr.columns()};
        expr.packet(p).copy_to(dst + i, Vc::flags::element_aligned);
    });

    for(int i = aligned_end; i < total_size; i++) {
        point_type p{i / expr.columns(), i % expr.columns()};
        dst[i] = expr[p];
    }
}

// -- evaluate a matrix expression by simply copying each value
template <typename E, typename T>
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expression, T* dst) {
    expression_member_t<decltype(simplify(expression))> expr = simplify(expression);

    algorithm::pfor(expr.size(), [&](const auto& pos) {
        int i = pos.x * expr.columns() + pos.y;
        dst[i] = expr[pos];
    });
}

template <typename T>
T conj(const T& x) {
    return x;
}

template <typename T>
std::complex<T> conj(const std::complex<T>& x) {
    return std::conj(x);
}

template <typename E>
auto eval(const MatrixExpression<E>& e) -> Matrix<scalar_type_t<E>> {
    using T = scalar_type_t<E>;
    Matrix<T> tmp(e.size());

    detail::evaluate(e, &tmp[{0, 0}]);

    return tmp;
}

template <typename T>
Matrix<T>& eval(Matrix<T>& m) {
    return m;
}

template <typename T>
const Matrix<T>& eval(const Matrix<T>& m) {
    return m;
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
    return sub<true>(e, {{r, 0}, {1, e.columns()}});
}

template <typename T, bool C>
auto row(const RefSubMatrix<T, C>& e, coordinate_type r) {
    assert_lt(r, e.rows());
    return sub<C>(e, {{r, 0}, {1, e.columns()}});
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

template <typename T>
auto column(const RefSubMatrix<T>& e, coordinate_type c) {
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
        return static_cast<const E&>(*this)[pos];
    }

    point_type size() const {
        return static_cast<const E&>(*this).size();
    }

    coordinate_type rows() const {
        return static_cast<const E&>(*this).rows();
    }

    coordinate_type columns() const {
        return static_cast<const E&>(*this).columns();
    }

    bool isSquare() const {
        return rows() == columns();
    }

    auto row(coordinate_type r) {
        return detail::row(static_cast<E&>(*this), r);
    }

    auto row(coordinate_type r) const {
        return detail::row(static_cast<const E&>(*this), r);
    }

    auto column(coordinate_type c) {
        return detail::column(static_cast<E&>(*this), c);
    }

    auto column(coordinate_type c) const {
        return detail::column(static_cast<const E&>(*this), c);
    }

    auto rowRange(range_type p) {
        return detail::sub<true>(static_cast<E&>(*this), {{p.x, 0}, {p.y, columns()}});
    }

    auto rowRange(range_type p) const {
        return detail::sub<true>(static_cast<const E&>(*this), {{p.x, 0}, {p.y, columns()}});
    }

    auto columnRange(range_type p) {
        return detail::sub<false>(static_cast<E&>(*this), {{0, p.x}, {rows(), p.y}});
    }

    auto columnRange(range_type p) const {
        return detail::sub<false>(static_cast<const E&>(*this), {{0, p.x}, {rows(), p.y}});
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
        return ElementMatrixMultiplication<E, E2>(static_cast<const E&>(*this), e);
    }

    MatrixTranspose<E> transpose() const {
        return MatrixTranspose<E>(static_cast<const E&>(*this));
    }

    MatrixConjugate<E> conjugate() const {
        return MatrixConjugate<E>(static_cast<const E&>(*this));
    }

    MatrixTranspose<MatrixConjugate<E>> adjoint() const {
        return this->conjugate().transpose();
    }

    auto sub(BlockRange block_range) {
        return detail::sub(static_cast<E&>(*this), block_range);
    }

    auto sub(BlockRange block_range) const {
        return detail::sub(static_cast<const E&>(*this), block_range);
    }

    MatrixAbs<E> abs() const {
        return MatrixAbs<E>(static_cast<const E&>(*this));
    }

    T norm() const {
        return std::sqrt(product(*this).reduce(0, std::plus<T>{}));
    }

    LUD<T> LUDecomposition() const {
        return LUD<T>(*this);
    }

    QRD<T> QRDecomposition() const {
        return QRD<T>(*this);
    }

    SVD<T> SVDecomposition() const {
        return SVD<T>(*this);
    }

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

    T determinant() const {
        return LUDecomposition().determinant();
    }

    Matrix<T> inverse() const {
        return LUDecomposition().inverse();
    }

    template <typename simd_type = PacketScalar>
    std::enable_if_t<vectorizable_v<E>, simd_type> packet(point_type p) const {
        return static_cast<const E&>(*this).template packet<simd_type>(p);
    }

    auto eval() {
        return detail::eval(static_cast<E&>(*this));
    }

    auto eval() const {
        return detail::eval(static_cast<const E&>(*this));
    }

    operator E&() {
        return static_cast<E&>(*this);
    }
    operator const E&() const {
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

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return lhs.template packet<simd_type>(p) + rhs.template packet<simd_type>(p);
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

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return lhs.template packet<simd_type>(p) - rhs.template packet<simd_type>(p);
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
    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return tmp.template packet<simd_type>(p);
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

        // TODO: pfor?
        for(ct i = 0; i < rows() - rows() % block_type::size()[0]; i += block_type::size()[0]) {
            for(ct j = 0; j < columns() - columns() % block_type::size()[1]; j += block_type::size()[1]) {
                block_type b(expression, {j, i});

                b.transpose();
                b.load_to(tmp, {i, j});
            }
        }


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

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return Vc::abs(expression.template packet<simd_type>(p));
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

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return expression.template packet<simd_type>(p) * simd_type(scalar);
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

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return simd_type(scalar) * expression.template packet<simd_type>(p);
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
        detail::evaluate(mat, &(*this)[{0, 0}]);
    }

    template <typename Derived>
    Matrix(const Eigen::MatrixBase<Derived>& matrix) : m_data({matrix.rows(), matrix.cols()}) {
        algorithm::pfor(size(), [&](const point_type& p) { m_data[p] = matrix(p.x, p.y); });
    }

    Matrix(const Matrix& mat) : MatrixExpression<Matrix<T>>(), m_data(mat.size()) {
        detail::evaluate(mat, &(*this)[{0, 0}]);
    }

    Matrix(Matrix&&) = default;

    Matrix& operator=(const Matrix& mat) {
        detail::evaluate(mat, &(*this)[{0, 0}]);

        return *this;
    }

    Matrix& operator=(Matrix&&) = default;

    Matrix& operator=(const T& value) {
        fill(value);
        return (*this);
    }

    template <typename E>
    Matrix& operator=(MatrixExpression<E> const& mat) {
        detail::evaluate(mat, &(*this)[{0, 0}]);

        return *this;
    }

    inline T& operator[](const point_type& pos) {
        return m_data[pos];
    }

    inline const T& operator[](const point_type& pos) const {
        return m_data[pos];
    }

    inline point_type size() const {
        return m_data.size();
    }

    inline coordinate_type rows() const {
        return m_data.size()[0];
    }

    inline coordinate_type columns() const {
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

    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p), Vc::flags::element_aligned);
    }

    const Matrix<T>& eval() const {
        return *this;
    }
    Matrix<T>& eval() {
        return *this;
    }

private:
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
    template <typename simd_type = PacketScalar>
    simd_type packet(point_type p) const {
        return simd_type(&operator[](p), Vc::flags::element_aligned);
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

template <typename T>
const Matrix<T>& simplify(const Matrix<T>& m) {
    return m;
}

template <typename T>
Matrix<T>& simplify(Matrix<T>& m) {
    return m;
}

template <typename T>
IdentityMatrix<T> simplify(IdentityMatrix<T> m) {
    return m;
}

template <typename T>
PermutationMatrix<T> simplify(PermutationMatrix<T> m) {
    return m;
}

template <typename T>
EvaluatedExpression<T> simplify(EvaluatedExpression<T> m) {
    return m;
}

template <typename E1, typename E2>
auto simplify(MatrixMultiplication<E1, E2> e) {
    Matrix<scalar_type_t<decltype(e)>> tmp(e.size());


    matrix_multiplication(tmp, simplify(e.getLeftExpression()), simplify(e.getRightExpression()));

    return std::move(EvaluatedExpression<scalar_type_t<decltype(e)>>(std::move(tmp)));
}

template <typename T>
const Matrix<T>& simplify(const MatrixExpression<Matrix<T>>& e) {
    return static_cast<const Matrix<T>&>(e);
}

template <typename E>
auto simplify(const MatrixExpression<E>& e) {
    return simplify(static_cast<const E&>(e));
}

template <typename E1, typename E2>
auto simplify(MatrixAddition<E1, E2> e) {
    return MatrixAddition<detail::remove_cvref_t<decltype(simplify(simplify(std::declval<E1>())))>,
                          detail::remove_cvref_t<decltype(simplify(std::declval<E2>()))>>(simplify(e.getLeftExpression()), simplify(e.getRightExpression()));
}

template <typename E1, typename E2>
auto simplify(MatrixSubtraction<E1, E2> e) {
    return MatrixSubtraction<detail::remove_cvref_t<decltype(simplify(simplify(std::declval<E1>())))>,
                             detail::remove_cvref_t<decltype(simplify(std::declval<E2>()))>>(simplify(e.getLeftExpression()), simplify(e.getRightExpression()));
}

template <typename E1, typename E2>
auto simplify(ElementMatrixMultiplication<E1, E2> e) {
    return ElementMatrixMultiplication<detail::remove_cvref_t<decltype(simplify(simplify(std::declval<E1>())))>,
                                       detail::remove_cvref_t<decltype(simplify(std::declval<E2>()))>>(simplify(e.getLeftExpression()),
                                                                                                       simplify(e.getRightExpression()));
}

template <typename E>
auto simplify(MatrixNegation<E> e) {
    return MatrixNegation<detail::remove_cvref_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()));
}

template <typename E>
std::enable_if_t<vectorizable_v<E>, EvaluatedExpression<scalar_type_t<MatrixTranspose<E>>>> simplify(MatrixTranspose<E> e) {
    Matrix<scalar_type_t<MatrixTranspose<E>>> tmp(e.size());

    e.evaluation(tmp);

    return std::move(EvaluatedExpression<scalar_type_t<decltype(e)>>(std::move(tmp)));
}

template <typename E>
std::enable_if_t<!vectorizable_v<E>, MatrixTranspose<E>> simplify(MatrixTranspose<E> e) {
    return e;
}

template <typename E>
auto simplify(MatrixConjugate<E> e) {
    return MatrixConjugate<detail::remove_cvref_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()));
}

template <typename E>
auto simplify(MatrixAbs<E> e) {
    return MatrixAbs<detail::remove_cvref_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()));
}

template <typename E, typename U>
auto simplify(MatrixScalarMultiplication<E, U> e) {
    return MatrixScalarMultiplication<detail::remove_cvref_t<decltype(simplify(std::declval<E>()))>, U>(simplify(e.getExpression()), e.getScalar());
}

template <typename E, typename U>
auto simplify(ScalarMatrixMultiplication<E, U> e) {
    return ScalarMatrixMultiplication<detail::remove_cvref_t<decltype(simplify(std::declval<E>()))>, U>(e.getScalar(), simplify(e.getExpression()));
}

template <typename E>
auto simplify(SubMatrix<E> e) {
    return SubMatrix<detail::remove_cvref_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()), e.getBlockRange());
}

template <typename T, bool C>
auto simplify(RefSubMatrix<T, C> e) {
    return e;
}

// What we really simplify
template <typename E1, typename E2>
auto simplify(SubMatrix<MatrixMultiplication<E1, E2>> e) {
    auto range = e.getBlockRange();
    BlockRange left({range.start.x, 0}, {range.size.x, e.getExpression().getLeftExpression().columns()});
    BlockRange right({0, range.start.y}, {e.getExpression().getRightExpression().rows(), range.size.y});

    return MatrixMultiplication<detail::remove_cvref_t<decltype(simplify(std::declval<SubMatrix<E1>>()))>,
                                detail::remove_cvref_t<decltype(simplify(std::declval<SubMatrix<E2>>()))>>(
        simplify(e.getExpression().getLeftExpression().sub(left)), simplify(e.getExpression().getRightExpression().sub(right)));
}

template <typename E>
expression_member_t<E> simplify(MatrixTranspose<MatrixTranspose<E>> e) {
    return e.getExpression().getExpression();
}

template <typename E>
expression_member_t<E> simplify(MatrixNegation<MatrixNegation<E>> e) {
    return e.getExpression().getExpression();
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, MatrixScalarMultiplication<E, U>>
simplify(MatrixScalarMultiplication<MatrixScalarMultiplication<E, U>, U> e) {
    return MatrixScalarMultiplication<E, U>(e.getExpression().getExpression(), e.getExpression().getScalar() * e.getScalar());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, ScalarMatrixMultiplication<E, U>>
simplify(ScalarMatrixMultiplication<MatrixScalarMultiplication<E, U>, U> e) {
    return ScalarMatrixMultiplication<E, U>(e.getExpression().getScalar() * e.getScalar(), e.getExpression().getExpression());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, ScalarMatrixMultiplication<E, U>>
simplify(ScalarMatrixMultiplication<ScalarMatrixMultiplication<E, U>, U> e) {
    return ScalarMatrixMultiplication<E, U>(e.getScalar() * e.getExpression().getScalar(), e.getExpression().getExpression());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, MatrixScalarMultiplication<E, U>>
simplify(MatrixScalarMultiplication<ScalarMatrixMultiplication<E, U>, U> e) {
    return MatrixScalarMultiplication<E, U>(e.getExpression().getExpression(), e.getExpression().getScalar() * e.getScalar());
}

template <typename E, typename T>
expression_member_t<E> simplify(MatrixMultiplication<E, IdentityMatrix<T>> e) {
    assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
    return e.getLeftExpression();
}

template <typename E, typename T>
expression_member_t<E> simplify(MatrixMultiplication<IdentityMatrix<T>, E> e) {
    assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
    return e.getRightExpression();
}

template <typename T>
IdentityMatrix<T> simplify(MatrixMultiplication<IdentityMatrix<T>, IdentityMatrix<T>> e) {
    assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
    return e.getLeftExpression();
}


namespace detail {

// fallback function
template <typename data_type>
void transpose(std::array<Vc::simd<data_type, Vc::simd_abi::scalar>, 1>&) {
    // Nothing to do in scalar case
}

#ifdef Vc_HAVE_SSE

// TODO

#endif

#ifdef Vc_HAVE_AVX

// -- AVX float 8x8
void transpose(std::array<Vc::simd<float, Vc::simd_abi::avx>, 8>& rows) {
    // TODO: check if this is valid
    __m256& r1 = reinterpret_cast<__m256&>(rows[0]);
    __m256& r2 = reinterpret_cast<__m256&>(rows[1]);
    __m256& r3 = reinterpret_cast<__m256&>(rows[2]);
    __m256& r4 = reinterpret_cast<__m256&>(rows[3]);
    __m256& r5 = reinterpret_cast<__m256&>(rows[4]);
    __m256& r6 = reinterpret_cast<__m256&>(rows[5]);
    __m256& r7 = reinterpret_cast<__m256&>(rows[6]);
    __m256& r8 = reinterpret_cast<__m256&>(rows[7]);

    __m256 t1, t2, t3, t4, t5, t6, t7, t8;
    __m256 u1, u2, u3, u4, u5, u6, u7, u8;


    t1 = _mm256_unpacklo_ps(r1, r2);
    t2 = _mm256_unpackhi_ps(r1, r2);
    t3 = _mm256_unpacklo_ps(r3, r4);
    t4 = _mm256_unpackhi_ps(r3, r4);
    t5 = _mm256_unpacklo_ps(r5, r6);
    t6 = _mm256_unpackhi_ps(r5, r6);
    t7 = _mm256_unpacklo_ps(r7, r8);
    t8 = _mm256_unpackhi_ps(r7, r8);

    u1 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    u3 = _mm256_shuffle_ps(t2, t4, _MM_SHUFFLE(1, 0, 1, 0));
    u4 = _mm256_shuffle_ps(t2, t4, _MM_SHUFFLE(3, 2, 3, 2));
    u5 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    u7 = _mm256_shuffle_ps(t6, t8, _MM_SHUFFLE(1, 0, 1, 0));
    u8 = _mm256_shuffle_ps(t6, t8, _MM_SHUFFLE(3, 2, 3, 2));


    r1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    r2 = _mm256_permute2f128_ps(u2, u6, 0x20);
    r3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    r4 = _mm256_permute2f128_ps(u4, u8, 0x20);
    r5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    r6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    r7 = _mm256_permute2f128_ps(u3, u7, 0x31);
    r8 = _mm256_permute2f128_ps(u4, u8, 0x31);
}

// -- AVX double 4x4
void transpose(std::array<Vc::simd<double, Vc::simd_abi::avx>, 4>& rows) {
    // TODO: check if this is valid
    __m256d& r1 = reinterpret_cast<__m256d&>(rows[0]);
    __m256d& r2 = reinterpret_cast<__m256d&>(rows[1]);
    __m256d& r3 = reinterpret_cast<__m256d&>(rows[2]);
    __m256d& r4 = reinterpret_cast<__m256d&>(rows[3]);

    __m256d t1, t2, t3, t4;

    t1 = _mm256_shuffle_pd(r1, r2, 0x0);
    t2 = _mm256_shuffle_pd(r3, r4, 0x0);
    t3 = _mm256_shuffle_pd(r1, r2, 0xF);
    t4 = _mm256_shuffle_pd(r3, r4, 0xF);

    r1 = _mm256_permute2f128_pd(t1, t2, 0x20);
    r2 = _mm256_permute2f128_pd(t3, t4, 0x20);
    r3 = _mm256_permute2f128_pd(t1, t2, 0x31);
    r4 = _mm256_permute2f128_pd(t3, t4, 0x31);
}


// -- AVX int 8x8
// -- TODO
// void transpose(std::array<Vc::simd<int, Vc::simd_abi::avx>, 8>& rows) {
// }

#endif

// TODO: move
template <typename Arg, typename _ = void>
struct transpose_exists : std::false_type {};

template <typename Arg>
struct transpose_exists<Arg, void_t<decltype(transpose(std::declval<Arg&>()))>> : std::true_type {};

template <typename Arg>
constexpr bool transpose_exists_v = transpose_exists<Arg>::value;

} // namespace detail

template <typename simd_type>
class SimdBlock {
    // TODO: decide abi tag is_vectorizable exp
    using T = typename simd_type::value_type;
    using abi_type =
        std::conditional_t<detail::transpose_exists_v<std::array<simd_type, simd_type::size()>>, typename simd_type::abi_type, Vc::simd_abi::scalar>;
    using simd_t = Vc::simd<T, abi_type>;

    static_assert(Vc::is_simd_v<simd_type>, "SimdBlock consists of SIMD vectors");

public:
    template <typename E>
    SimdBlock(const MatrixExpression<E>& exp, point_type pos) {
        for(coordinate_type i = 0; i < (coordinate_type)simd_t::size(); ++i) {
            rows[i] = exp.template packet<simd_t>({pos.x + i, pos.y});
        }
    }

    static point_type size() {
        return {simd_t::size(), simd_t::size()};
    }

    void transpose() {
        detail::transpose(rows);
    }

    void load_to(Matrix<T>& matrix, point_type pos) {
        for(coordinate_type i = 0; i < (coordinate_type)simd_t::size(); ++i) {
            rows[i].copy_to(&matrix[{pos.x + i, pos.y}], Vc::flags::element_aligned);
        }
    }

private:
    std::array<simd_t, simd_t::size()> rows;
};

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
