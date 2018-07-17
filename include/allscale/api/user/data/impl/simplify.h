#pragma once


#include "allscale/api/user/data/impl/evaluate.h"
#include "allscale/api/user/data/impl/expressions.h"
#include "allscale/api/user/data/impl/matrix_multiplication.h"
#include "allscale/api/user/data/impl/traits.h"
#include "allscale/api/user/data/impl/types.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

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

template <typename E1, typename E2>
Matrix<scalar_type_t<MatrixMultiplication<E1, E2>>>& simplify(MatrixMultiplication<E1, E2> e, Matrix<scalar_type_t<MatrixMultiplication<E1, E2>>>& m) {
    assert_eq(e.size(), m.size());

    matrix_multiplication(m, simplify(e.getLeftExpression()), simplify(e.getRightExpression()));

    return m;
}

template <typename E>
auto simplify(const MatrixExpression<E>& e, Matrix<scalar_type_t<E>>& dst) {
    const E& mexpr = static_cast<const E&>(e);
    expression_member_t<decltype(simplify(mexpr))> exp = simplify(mexpr);
    detail::evaluate(exp, dst);
}

template <typename E>
auto simplify(const MatrixExpression<E>& e, SubMatrix<Matrix<scalar_type_t<E>>> dst) {
    const E& mexpr = static_cast<const E&>(e);
    expression_member_t<decltype(simplify(mexpr))> exp = simplify(mexpr);

    detail::evaluate(exp, dst);
}

template <typename E1, typename E2>
auto simplify(MatrixAddition<E1, E2> e) {
    return MatrixAddition<std::remove_reference_t<decltype(simplify(simplify(std::declval<E1>())))>,
                          std::remove_reference_t<decltype(simplify(std::declval<E2>()))>>(simplify(e.getLeftExpression()), simplify(e.getRightExpression()));
}

template <typename E1, typename E2>
auto simplify(MatrixSubtraction<E1, E2> e) {
    return MatrixSubtraction<std::remove_reference_t<decltype(simplify(simplify(std::declval<E1>())))>,
                             std::remove_reference_t<decltype(simplify(std::declval<E2>()))>>(simplify(e.getLeftExpression()),
                                                                                              simplify(e.getRightExpression()));
}

template <typename E1, typename E2>
auto simplify(ElementMatrixMultiplication<E1, E2> e) {
    return ElementMatrixMultiplication<std::remove_reference_t<decltype(simplify(simplify(std::declval<E1>())))>,
                                       std::remove_reference_t<decltype(simplify(std::declval<E2>()))>>(simplify(e.getLeftExpression()),
                                                                                                        simplify(e.getRightExpression()));
}

template <typename E>
auto simplify(MatrixNegation<E> e) {
    return MatrixNegation<std::remove_reference_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()));
}

template <typename E>
std::enable_if_t<vectorizable_v<E>, EvaluatedExpression<scalar_type_t<MatrixTranspose<E>>>> simplify(MatrixTranspose<E> e) {
    Matrix<scalar_type_t<MatrixTranspose<E>>> tmp(e.size());

    e.evaluation(tmp);

    return std::move(EvaluatedExpression<scalar_type_t<decltype(e)>>(std::move(tmp)));
}

template <typename E>
std::enable_if_t<!vectorizable_v<E>, EvaluatedExpression<scalar_type_t<MatrixTranspose<E>>>> simplify(MatrixTranspose<E> e) {
    Matrix<scalar_type_t<MatrixTranspose<E>>> tmp(e.size());

    algorithm::pfor(e.size(), [&](const auto& pos) { tmp[pos] = e[pos]; });

    return std::move(EvaluatedExpression<scalar_type_t<decltype(e)>>(std::move(tmp)));
}

template <typename E>
auto simplify(MatrixConjugate<E> e) {
    return MatrixConjugate<std::remove_reference_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()));
}

template <typename E>
auto simplify(MatrixAbs<E> e) {
    return MatrixAbs<std::remove_reference_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()));
}

template <typename E, typename U>
auto simplify(MatrixScalarMultiplication<E, U> e) {
    return MatrixScalarMultiplication<std::remove_reference_t<decltype(simplify(std::declval<E>()))>, U>(simplify(e.getExpression()), e.getScalar());
}

template <typename E, typename U>
auto simplify(ScalarMatrixMultiplication<E, U> e) {
    return ScalarMatrixMultiplication<std::remove_reference_t<decltype(simplify(std::declval<E>()))>, U>(e.getScalar(), simplify(e.getExpression()));
}

template <typename E>
auto simplify(SubMatrix<E> e) {
    return SubMatrix<std::remove_reference_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()), e.getBlockRange());
}

// What we really simplify
template <typename E1, typename E2>
auto simplify(SubMatrix<MatrixMultiplication<E1, E2>> e) {
    auto range = e.getBlockRange();
    BlockRange left({range.start.x, 0}, {range.size.x, e.getExpression().getLeftExpression().columns()});
    BlockRange right({0, range.start.y}, {e.getExpression().getRightExpression().rows(), range.size.y});

    return MatrixMultiplication<std::remove_reference_t<decltype(simplify(std::declval<SubMatrix<E1>>()))>,
                                std::remove_reference_t<decltype(simplify(std::declval<SubMatrix<E2>>()))>>(
        simplify(e.getExpression().getLeftExpression().sub(left)), simplify(e.getExpression().getRightExpression().sub(right)));
}

template <typename E>
expression_member_t<E>& simplify(MatrixTranspose<MatrixTranspose<E>> e) {
    return e.getExpression().getExpression();
}

template <typename E>
expression_member_t<E>& simplify(MatrixNegation<MatrixNegation<E>> e) {
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

template <typename E>
void evaluate_simplify(const MatrixExpression<E>& expression, Matrix<scalar_type_t<E>>& dst) {
    assert_eq(expression.size(), dst.size());

    simplify(static_cast<const E&>(expression), dst);
}

template <typename E>
void evaluate_simplify(const MatrixExpression<E>& expression, SubMatrix<Matrix<scalar_type_t<E>>> dst) {
    assert_eq(expression.size(), dst.size());

    simplify(static_cast<const E&>(expression), dst);
}

template <typename E>
auto eval(const MatrixExpression<E>& e) -> Matrix<scalar_type_t<E>> {
    using T = scalar_type_t<E>;
    Matrix<T> tmp(e.size());

    detail::evaluate_simplify(e, tmp);

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

} // namespace detail


template <typename E>
auto MatrixExpression<E>::eval() -> detail::eval_return_t<std::remove_reference_t<decltype(impl())>> {
    return detail::eval(impl());
}

template <typename E>
auto MatrixExpression<E>::eval() const -> detail::eval_return_t<std::remove_reference_t<decltype(impl())>> {
    return detail::eval(impl());
}

template <typename T>
template <typename E>
void Matrix<T>::evaluate(const MatrixExpression<E>& mat) {
    detail::evaluate_simplify(mat, *this);
}

template <typename T>
template <typename E>
void SubMatrix<Matrix<T>>::evaluate(const MatrixExpression<E>& mat) {
    detail::evaluate_simplify(mat, *this);
}

template <typename T>
template <typename T2>
void SubMatrix<Matrix<T>>::swap(SubMatrix<Matrix<T2>> other) {
    assert_eq(size(), other.size());
    detail::swap(*this, other);
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
