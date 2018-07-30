#pragma once

#include <Vc/Vc>
#include <cstddef>
#include <functional>
#include <type_traits>

#include "allscale/api/user/data/impl/forward.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

namespace detail {

template <typename T>
struct set_type {
    using type = T;
};

template <bool... A>
struct and_value;

template <bool A>
struct and_value<A> {
    static constexpr bool value = A;
};

template <bool A, bool... B>
struct and_value<A, B...> {
    static constexpr bool value = A && and_value<B...>::value;
};

// C++ 17 feature
template <class...>
using void_t = void;

// C++ 20 feature
template <typename T>
struct remove_cvref : public set_type<std::remove_cv_t<std::remove_reference_t<T>>> {};

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <typename T>
struct alignment
    : public std::conditional<alignof(std::max_align_t) >= Vc::memory_alignment_v<T>, Vc::flags::vector_aligned_tag, Vc::flags::element_aligned_tag> {};

template <typename T>
using alignment_t = typename alignment<T>::type;

} // namespace detail

template <typename Functor, typename T1, typename T2>
struct operation_result : public detail::set_type<decltype(std::declval<Functor>()(std::declval<T1>(), std::declval<T2>()))> {};

template <typename Functor, typename T1, typename T2>
using operation_result_t = typename operation_result<Functor, T1, T2>::type;

template <typename E>
struct expression_traits;

template <typename T>
struct expression_traits<Matrix<T>> {
    // -- types
    using scalar_type = T;
    using eval_return_type = Matrix<T>&;
    using expression_member_type = Matrix<T>&;
    using expression_tree_type = Matrix<T>;

    // -- values
    static constexpr bool vectorizable = std::is_arithmetic<T>::value;
};

template <typename T>
struct expression_traits<const Matrix<T>> {
    // -- types
    using scalar_type = T;
    using eval_return_type = const Matrix<T>&;
    using expression_member_type = const Matrix<T>&;
    using expression_tree_type = const Matrix<T>;

    // -- values
    static constexpr bool vectorizable = std::is_arithmetic<T>::value;
};

template <typename E1, typename E2>
struct expression_traits<MatrixAddition<E1, E2>> {
private:
    using expr1_t = expression_traits<E1>;
    using expr2_t = expression_traits<E2>;

public:
    // -- types
    using scalar_type = operation_result_t<std::plus<>, typename expr1_t::scalar_type, typename expr2_t::scalar_type>;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixAddition<E1, E2>;
    using expression_tree_type = MatrixAddition<E1, E2>;

    // -- values
    static constexpr bool vectorizable =
        expr1_t::vectorizable && expr2_t::vectorizable && std::is_same<typename expr1_t::scalar_type, typename expr2_t::scalar_type>::value;
};

template <typename E1, typename E2>
struct expression_traits<MatrixSubtraction<E1, E2>> {
private:
    using expr1_t = expression_traits<E1>;
    using expr2_t = expression_traits<E2>;

public:
    // -- types
    using scalar_type = operation_result_t<std::minus<>, typename expr1_t::scalar_type, typename expr2_t::scalar_type>;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixSubtraction<E1, E2>;
    using expression_tree_type = MatrixSubtraction<E1, E2>;

    // -- values
    static constexpr bool vectorizable =
        expr1_t::vectorizable && expr2_t::vectorizable && std::is_same<typename expr1_t::scalar_type, typename expr2_t::scalar_type>::value;
};

template <typename E1, typename E2>
struct expression_traits<ElementMatrixMultiplication<E1, E2>> {
private:
    using expr1_t = expression_traits<E1>;
    using expr2_t = expression_traits<E2>;

public:
    // -- types
    using scalar_type = operation_result_t<std::multiplies<>, typename expr1_t::scalar_type, typename expr2_t::scalar_type>;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = ElementMatrixMultiplication<E1, E2>;
    using expression_tree_type = ElementMatrixMultiplication<E1, E2>;

    // -- values
    static constexpr bool vectorizable =
        expr1_t::vectorizable && expr2_t::vectorizable && std::is_same<typename expr1_t::scalar_type, typename expr2_t::scalar_type>::value;
};

template <typename E1, typename E2>
struct expression_traits<MatrixMultiplication<E1, E2>> {
private:
    using expr1_t = expression_traits<E1>;
    using expr2_t = expression_traits<E2>;

public:
    // -- types
    using scalar_type = operation_result_t<std::multiplies<>, typename expr1_t::scalar_type, typename expr2_t::scalar_type>;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixMultiplication<E1, E2>;
    using expression_tree_type = MatrixMultiplication<E1, E2>;

    // -- values
    static constexpr bool vectorizable = false;

    static_assert(std::is_same<operation_result_t<std::plus<>, scalar_type, scalar_type>, scalar_type>::value,
                  "Resulting type of matrix multiplication must yield the same type if added up.");
};

template <typename T>
struct expression_traits<EvaluatedExpression<T>> {
    // -- types
    using scalar_type = T;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = EvaluatedExpression<T>;
    using expression_tree_type = EvaluatedExpression<T>;

    // -- values
    static constexpr bool vectorizable = std::is_arithmetic<scalar_type>::value;
};

template <typename E>
struct expression_traits<MatrixNegation<E>> {
private:
    using expr_t = expression_traits<E>;

public:
    // -- types
    using scalar_type = typename expr_t::scalar_type;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixNegation<E>;
    using expression_tree_type = MatrixNegation<E>;

    // -- values
    static constexpr bool vectorizable = expr_t::vectorizable;
};

template <typename E>
struct expression_traits<MatrixTranspose<E>> {
private:
    using expr_t = expression_traits<E>;

public:
    // -- types
    using scalar_type = typename expr_t::scalar_type;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixTranspose<E>;
    using expression_tree_type = MatrixTranspose<E>;

    // -- values
    static constexpr bool vectorizable = false;
};

template <typename E>
struct expression_traits<MatrixConjugate<E>> {
private:
    using expr_t = expression_traits<E>;

public:
    // -- types
    using scalar_type = typename expr_t::scalar_type;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixConjugate<E>;
    using expression_tree_type = MatrixConjugate<E>;

    // -- values
    static constexpr bool vectorizable = false;
};

template <typename E>
struct expression_traits<MatrixAbs<E>> {
private:
    using expr_t = expression_traits<E>;

public:
    // -- types
    using scalar_type = typename expr_t::scalar_type;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixAbs<E>;
    using expression_tree_type = MatrixAbs<E>;

    // -- values
    static constexpr bool vectorizable = expr_t::vectorizable;
};

template <typename E, typename U>
struct expression_traits<MatrixScalarMultiplication<E, U>> {
private:
    using expr_t = expression_traits<E>;

public:
    // -- types
    using scalar_type = operation_result_t<std::multiplies<>, typename expr_t::scalar_type, U>;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixScalarMultiplication<E, U>;
    using expression_tree_type = MatrixScalarMultiplication<E, U>;

    // -- values
    static constexpr bool vectorizable = expr_t::vectorizable && std::is_same<typename expr_t::scalar_type, U>::value;
};

template <typename E, typename U>
struct expression_traits<ScalarMatrixMultiplication<E, U>> {
private:
    using expr_t = expression_traits<E>;

public:
    // -- types
    using scalar_type = operation_result_t<std::multiplies<>, U, typename expr_t::scalar_type>;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = ScalarMatrixMultiplication<E, U>;
    using expression_tree_type = ScalarMatrixMultiplication<E, U>;

    // -- values
    static constexpr bool vectorizable = expr_t::vectorizable && std::is_same<typename expr_t::scalar_type, U>::value;
};

template <typename T>
struct expression_traits<IdentityMatrix<T>> {
    // -- types
    using scalar_type = T;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = IdentityMatrix<T>;
    using expression_tree_type = IdentityMatrix<T>;

    // -- values
    static constexpr bool vectorizable = false;
};

template <typename T>
struct expression_traits<PermutationMatrix<T>> {
    // -- types
    using scalar_type = T;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = PermutationMatrix<T>;
    using expression_tree_type = PermutationMatrix<T>;

    // -- values
    static constexpr bool vectorizable = false;
};

template <typename E, ViewType View>
struct expression_traits<MatrixView<E, View>> {
private:
    using expr_t = expression_traits<E>;

public:
    // -- types
    using scalar_type = typename expr_t::scalar_type;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = MatrixView<E, View>;
    using expression_tree_type = MatrixView<E, View>;

    // -- values
    static constexpr bool vectorizable = false;
};

// template <typename E, ViewType View>
// struct expression_traits<MatrixView<E, View>> {
// private:
//    using expr_t = expression_traits<E>;
//
// public:
//    // -- types
//    using scalar_type = typename expr_t::scalar_type;
//    using eval_return_type = Matrix<scalar_type>;
//    using expression_member_type = MatrixView<E, View>;
//    using expression_tree_type = MatrixView<E, View>;
//
//    // -- values
//    static constexpr bool vectorizable = false;
//};

template <typename E, bool V>
struct expression_traits<SubMatrix<E, V>> {
private:
    using expr_t = expression_traits<E>;

public:
    // -- types
    using scalar_type = typename expr_t::scalar_type;
    using eval_return_type = Matrix<scalar_type>;
    using expression_member_type = SubMatrix<E, V>;
    using expression_tree_type = SubMatrix<E, V>;

    // -- values
    static constexpr bool vectorizable = expr_t::vectorizable;
};

template <typename E>
struct expression_traits<const MatrixExpression<E>> : public expression_traits<MatrixExpression<E>> {};

template <typename E>
struct expression_traits<const E> : public expression_traits<E> {};

template <typename E>
struct expression_traits<volatile E> : public expression_traits<E> {};

template <typename E>
struct expression_traits<const volatile E> : public expression_traits<E> {};

template <typename E>
struct expression_traits<E&> : public expression_traits<E> {};

template <typename E>
using eval_return_t = typename expression_traits<E>::eval_return_type;

template <typename E>
using scalar_type_t = typename expression_traits<E>::scalar_type;

template <typename E>
using eval_return_t = typename expression_traits<E>::eval_return_type;

template <typename E>
using expression_member_t = typename expression_traits<E>::expression_member_type;

template <typename E>
using expression_tree_t = typename expression_traits<E>::expression_tree_type;

template <typename E>
static constexpr bool vectorizable_v = expression_traits<E>::vectorizable;

template <typename T>
struct is_associative : public std::false_type {};

template <>
struct is_associative<int> : public std::true_type {};

template <>
struct is_associative<unsigned> : public std::true_type {};

template <>
struct is_associative<long> : public std::true_type {};

template <>
struct is_associative<unsigned long> : public std::true_type {};

#ifndef ALLSCALE_NO_FAST_MATH

template <>
struct is_associative<double> : public std::true_type {};

template <>
struct is_associative<float> : public std::true_type {};

#endif

template <typename T>
constexpr bool is_associative_v = is_associative<T>::value;

// -- checks if expression allows direct matrix access
template <typename E>
struct direct_access : public std::false_type {};

template <typename T>
struct direct_access<Matrix<T>> : public std::true_type {};

template <typename T, bool V>
struct direct_access<SubMatrix<Matrix<T>, V>> : public std::true_type {};

template <typename T, bool V>
struct direct_access<SubMatrix<const Matrix<T>, V>> : public std::true_type {};

template <typename E>
constexpr bool direct_access_v = direct_access<E>::value;

// -- checks if the expression is a transpose of a matrix
template <typename E>
struct is_transpose : public std::false_type {};

template <typename E>
struct is_transpose<MatrixTranspose<E>> : public direct_access<E> {};

template <typename E>
constexpr bool is_transpose_v = is_transpose<E>::value;

template <typename E>
constexpr bool direct_or_transpose_v = direct_access_v<E> || is_transpose_v<E>;

template <typename F, typename T>
struct type_consistent : public std::is_same<T, operation_result_t<F, T, T>> {};

template <typename F, typename T>
constexpr bool type_consistent_v = type_consistent<F, T>::value;

template <typename F, typename E1, typename E2, class = void>
struct is_valid : public std::false_type {};

// TODO: use operation_result_t<F, E1, E2> here
template <typename F, typename E1, typename E2>
struct is_valid<F, E1, E2, detail::void_t<decltype(std::declval<F>()(std::declval<E1>(), std::declval<E2>()))>> : public std::true_type {};

template <typename F, typename E1, typename E2>
constexpr bool is_valid_v = is_valid<F, E1, E2>::value;

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
