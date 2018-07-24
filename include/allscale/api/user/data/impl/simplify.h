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


// -- base cases
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

// -- simplify subexpression / don't touch current expression

template <typename E1, typename E2>
auto simplify(MatrixMultiplication<SubMatrix<E1, true>, SubMatrix<E2, true>> e) {
    return simplify(e.getLeftExpression()) * simplify(e.getRightExpression());
}

template <typename E1, typename E2>
auto simplify(MatrixAddition<E1, E2> e) {
    return simplify(e.getLeftExpression()) + simplify(e.getRightExpression());
}

template <typename E1, typename E2>
auto simplify(MatrixSubtraction<E1, E2> e) {
    return simplify(e.getLeftExpression()) - simplify(e.getRightExpression());
}

template <typename E1, typename E2>
auto simplify(ElementMatrixMultiplication<E1, E2> e) {
    return simplify(e.getLeftExpression()).product(simplify(e.getRightExpression()));
}

template <typename E>
auto simplify(MatrixNegation<E> e) {
    return -simplify(e.getExpression());
}

template <typename E, ViewType View>
auto simplify(MatrixView<E, View> e) {
    return simplify(e.getExpression()).template view<View>();
}

template <typename E>
auto simplify(MatrixConjugate<E> e) {
    return simplify(e.getExpression()).conjugate();
}

template <typename E>
auto simplify(MatrixAbs<E> e) {
    return simplify(e.getExpression()).abs();
}

template <typename E, typename U>
auto simplify(MatrixScalarMultiplication<E, U> e) {
    return simplify(e.getExpression()) * e.getScalar();
}

template <typename E, typename U>
auto simplify(ScalarMatrixMultiplication<E, U> e) {
    return e.getScalar() * simplify(e.getExpression());
}

template <typename E, bool V>
auto simplify(SubMatrix<E, V> e) {
    return simplify(e.getExpression()).sub(e.getBlockRange());
}

// -- create temporaries for these expressions
template <typename E1, typename E2>
auto simplify(MatrixMultiplication<E1, E2> e) {
    Matrix<scalar_type_t<decltype(e)>> tmp(e.size());

    matrix_multiplication(tmp, simplify(e.getLeftExpression()), simplify(e.getRightExpression()));

    return std::move(EvaluatedExpression<scalar_type_t<decltype(e)>>(std::move(tmp)));
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

// -- optimizations
template <typename E1, typename E2>
auto simplify(SubMatrix<MatrixMultiplication<E1, E2>> e) {
    auto range = e.getBlockRange();
    BlockRange left({range.start.x, 0}, {range.size.x, e.getExpression().getLeftExpression().columns()});
    BlockRange right({0, range.start.y}, {e.getExpression().getRightExpression().rows(), range.size.y});

    return simplify(simplify(e.getExpression().getLeftExpression().sub(left)) * simplify(e.getExpression().getRightExpression().sub(right)));
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
    return simplify(e.getExpression().getExpression() * (e.getExpression().getScalar() * e.getScalar()));
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, ScalarMatrixMultiplication<E, U>>
simplify(ScalarMatrixMultiplication<MatrixScalarMultiplication<E, U>, U> e) {
    return simplify((e.getExpression().getScalar() * e.getScalar()) * e.getExpression().getExpression());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, ScalarMatrixMultiplication<E, U>>
simplify(ScalarMatrixMultiplication<ScalarMatrixMultiplication<E, U>, U> e) {
    return simplify((e.getScalar() * e.getExpression().getScalar()) * e.getExpression().getExpression());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_v<std::multiplies<>, U>, MatrixScalarMultiplication<E, U>>
simplify(MatrixScalarMultiplication<ScalarMatrixMultiplication<E, U>, U> e) {
    return simplify(e.getExpression().getExpression() * (e.getExpression().getScalar() * e.getScalar()));
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
auto MatrixExpression<E>::eval() -> detail::eval_return_t<std::remove_reference_t<decltype(impl())>> {
    return detail::eval(impl());
}

template <typename E>
auto MatrixExpression<E>::eval() const -> detail::eval_return_t<std::remove_reference_t<decltype(impl())>> {
    return detail::eval(impl());
}

template <typename E>
template <typename E2>
void AccessBase<E>::evaluate(const MatrixExpression<E2>& mat) {
    detail::evaluate_simplify(mat, static_cast<E&>(*this));
}


template <typename T, bool V>
template <typename T2, bool V2>
void SubMatrix<Matrix<T>, V>::swap(SubMatrix<Matrix<T2>, V2> other) {
    assert_eq(size(), other.size());
    detail::swap(*this, other);
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
