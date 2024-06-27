#pragma once


#include "bla/impl/evaluate.h"
#include "bla/impl/expressions.h"
#include "bla/impl/matrix_multiplication.h"
#include "bla/impl/traits.h"
#include "bla/impl/types.h"

namespace bla {
namespace impl {


// -- base cases
template <typename T>
const Matrix<T>& simplify_step(const Matrix<T>& m) {
    return m;
}

template <typename T>
Matrix<T>& simplify_step(Matrix<T>& m) {
    return m;
}

template <typename T>
IdentityMatrix<T> simplify_step(IdentityMatrix<T> m) {
    return m;
}

template <typename T>
PermutationMatrix<T> simplify_step(PermutationMatrix<T> m) {
    return m;
}

template <typename T>
EvaluatedExpression<T> simplify_step(EvaluatedExpression<T> m) {
    return m;
}

// -- simplify subexpression / don't touch current expression

template <typename E1, typename E2>
auto simplify_step(MatrixAddition<E1, E2> e) {
    return simplify_step(e.getLeftExpression()) + simplify_step(e.getRightExpression());
}

template <typename E1, typename E2>
auto simplify_step(MatrixSubtraction<E1, E2> e) {
    return simplify_step(e.getLeftExpression()) - simplify_step(e.getRightExpression());
}

template <typename E1, typename E2>
auto simplify_step(ElementMatrixMultiplication<E1, E2> e) {
    return simplify_step(e.getLeftExpression()).product(simplify_step(e.getRightExpression()));
}

template <typename E>
auto simplify_step(MatrixNegation<E> e) {
    return -simplify_step(e.getExpression());
}

template <typename E, ViewType View>
auto simplify_step(MatrixView<E, View> e) {
    return simplify_step(e.getExpression()).template view<View>();
}

template <typename E>
auto simplify_step(MatrixConjugate<E> e) {
    return simplify_step(e.getExpression()).conjugate();
}

template <typename E>
auto simplify_step(MatrixAbs<E> e) {
    return simplify_step(e.getExpression()).abs();
}

template <typename E, typename U>
auto simplify_step(MatrixScalarMultiplication<E, U> e) {
    return simplify_step(e.getExpression()) * e.getScalar();
}

template <typename E, typename U>
auto simplify_step(ScalarMatrixMultiplication<E, U> e) {
    return e.getScalar() * simplify_step(e.getExpression());
}

template <typename E>
auto simplify_step(SubMatrix<E> e) {
    return simplify_step(e.getExpression()).sub(e.getBlockRange());
}

// -- create temporaries for these expressions
template <typename E1, typename E2>
auto simplify_step(MatrixMultiplication<E1, E2> e) {
    Matrix<scalar_type_t<decltype(e)>> tmp(e.size());

    matrix_multiplication(tmp, simplify_step(e.getLeftExpression()), simplify_step(e.getRightExpression()));

    return EvaluatedExpression<scalar_type_t<decltype(e)>>(std::move(tmp));
}

template <typename E>
std::enable_if_t<vectorizable_v<E>, EvaluatedExpression<scalar_type_t<MatrixTranspose<E>>>> simplify_step(MatrixTranspose<E> e) {
    Matrix<scalar_type_t<MatrixTranspose<E>>> tmp(e.size());

    e.evaluation(tmp);

    return EvaluatedExpression<scalar_type_t<decltype(e)>>(std::move(tmp));
}

template <typename E>
std::enable_if_t<!vectorizable_v<E>, EvaluatedExpression<scalar_type_t<MatrixTranspose<E>>>> simplify_step(MatrixTranspose<E> e) {
    Matrix<scalar_type_t<MatrixTranspose<E>>> tmp(e.size());

    allscale::api::user::algorithm::pfor(e.size(), [&](const auto& pos) { tmp[pos] = e[pos]; });

    return EvaluatedExpression<scalar_type_t<decltype(e)>>(std::move(tmp));
}

// -- optimizations
template <typename E1, typename E2>
auto simplify_step(SubMatrix<MatrixMultiplication<E1, E2>> e) {
    auto range = e.getBlockRange();
    BlockRange left({range.start.x, 0}, {range.size.x, e.getExpression().getLeftExpression().columns()});
    BlockRange right({0, range.start.y}, {e.getExpression().getRightExpression().rows(), range.size.y});

    return simplify_step(simplify_step(e.getExpression().getLeftExpression().sub(left)) * simplify_step(e.getExpression().getRightExpression().sub(right)));
}

template <typename E1, typename E2>
auto simplify_step(SubMatrix<MatrixAddition<E1, E2>> e) {
    return simplify_step(simplify_step(e.getExpression().getLeftExpression().sub(e.getBlockRange()))
                         + simplify_step(e.getExpression().getRightExpression().sub(e.getBlockRange())));
}

template <typename E1, typename E2>
auto simplify_step(SubMatrix<MatrixSubtraction<E1, E2>> e) {
    return simplify_step(simplify_step(e.getExpression().getLeftExpression().sub(e.getBlockRange()))
                         - simplify_step(e.getExpression().getRightExpression().sub(e.getBlockRange())));
}

template <typename E>
expression_member_t<E> simplify_step(MatrixTranspose<MatrixTranspose<E>> e) {
    return e.getExpression().getExpression();
}

template <typename E>
expression_member_t<E> simplify_step(MatrixNegation<MatrixNegation<E>> e) {
    return e.getExpression().getExpression();
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, MatrixScalarMultiplication<E, U>>
simplify_step(MatrixScalarMultiplication<MatrixScalarMultiplication<E, U>, U> e) {
    return simplify_step(e.getExpression().getExpression() * (e.getExpression().getScalar() * e.getScalar()));
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, ScalarMatrixMultiplication<E, U>>
simplify_step(ScalarMatrixMultiplication<MatrixScalarMultiplication<E, U>, U> e) {
    return simplify_step((e.getExpression().getScalar() * e.getScalar()) * e.getExpression().getExpression());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, ScalarMatrixMultiplication<E, U>>
simplify_step(ScalarMatrixMultiplication<ScalarMatrixMultiplication<E, U>, U> e) {
    return simplify_step((e.getScalar() * e.getExpression().getScalar()) * e.getExpression().getExpression());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, MatrixScalarMultiplication<E, U>>
simplify_step(MatrixScalarMultiplication<ScalarMatrixMultiplication<E, U>, U> e) {
    return simplify_step(e.getExpression().getExpression() * (e.getExpression().getScalar() * e.getScalar()));
}

template <typename E, typename T>
expression_member_t<E> simplify_step(MatrixMultiplication<E, IdentityMatrix<T>> e) {
    assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
    return e.getLeftExpression();
}

template <typename E, typename T>
expression_member_t<E> simplify_step(MatrixMultiplication<IdentityMatrix<T>, E> e) {
    assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
    return e.getRightExpression();
}

template <typename T>
IdentityMatrix<T> simplify_step(MatrixMultiplication<IdentityMatrix<T>, IdentityMatrix<T>> e) {
    assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
    return e.getLeftExpression();
}

#ifndef BLA_NO_ETO

template <typename E>
auto simplify(E&& e) -> decltype(simplify_step(std::forward<E>(e))) {
    return simplify_step(std::forward<E>(e));
}

#else

template <typename E>
auto simplify(E&& e) -> decltype(std::forward<E>(e)) {
    return std::forward<E>(e);
}

#endif

// -- optimize no alias
template <typename E1, typename E2>
Matrix<scalar_type_t<MatrixMultiplication<E1, E2>>>& noalias_evaluate(MatrixMultiplication<E1, E2> e, Matrix<scalar_type_t<MatrixMultiplication<E1, E2>>>& m) {
    assert_eq(e.size(), m.size());

    matrix_multiplication(m, simplify(e.getLeftExpression()), simplify(e.getRightExpression()));

    return m;
}

template <typename E>
auto noalias_evaluate(const MatrixExpression<E>& e, Matrix<scalar_type_t<E>>& dst) {
    const E& mexpr = static_cast<const E&>(e);
    expression_member_t<decltype(simplify(mexpr))> exp = simplify(mexpr);
    detail::evaluate(exp, dst);
}

template <typename E>
auto noalias_evaluate(const MatrixExpression<E>& e, SubMatrix<Matrix<scalar_type_t<E>>> dst) {
    const E& mexpr = static_cast<const E&>(e);
    expression_member_t<decltype(simplify(mexpr))> exp = simplify(mexpr);

    detail::evaluate(exp, dst);
}

namespace detail {

template <typename E>
void evaluate_simplify(const MatrixExpression<E>& expression, Matrix<scalar_type_t<E>>& dst) {
    assert_eq(expression.size(), dst.size());

    noalias_evaluate(static_cast<const E&>(expression), dst);
}

template <typename E>
void evaluate_simplify(const MatrixExpression<E>& expression, SubMatrix<Matrix<scalar_type_t<E>>> dst) {
    assert_eq(expression.size(), dst.size());

    noalias_evaluate(static_cast<const E&>(expression), dst);
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
auto MatrixExpression<E>::eval() -> eval_return_t<std::remove_reference_t<decltype(impl())>> {
    return detail::eval(impl());
}

template <typename E>
auto MatrixExpression<E>::eval() const -> eval_return_t<std::remove_reference_t<decltype(impl())>> {
    return detail::eval(impl());
}

template <typename E>
template <typename E2>
void AccessBase<E>::evaluate(const MatrixExpression<E2>& mat) {
    detail::evaluate_simplify(mat, static_cast<E&>(*this));
}


template <typename T>
template <typename T2>
void SubMatrix<Matrix<T>>::swap(SubMatrix<Matrix<T2>> other) {
    assert_eq(size(), other.size());
    detail::swap(*this, other);
}

} // end namespace impl
} // namespace bla
