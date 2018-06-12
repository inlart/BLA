#pragma once

#include "allscale/api/user/data/impl/expressions.h"

#include "allscale/api/user/data/impl/forward.h"

#include <functional>
#include <type_traits>

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

} // end namespace detail


template <typename Functor, typename T1, typename T2>
struct operation_result : public detail::set_type<decltype(std::declval<Functor>()(std::declval<T1>(), std::declval<T2>()))> {};

template <typename Functor, typename T1, typename T2>
using operation_result_t = typename operation_result<Functor, T1, T2>::type;

template <typename Expr>
struct scalar_type;

template <typename Expr>
struct scalar_type<const Expr> : public detail::set_type<typename scalar_type<Expr>::type> {};

template <typename Expr>
struct scalar_type<volatile Expr> : public detail::set_type<typename scalar_type<Expr>::type> {};

template <typename Expr>
struct scalar_type<const volatile Expr> : public detail::set_type<typename scalar_type<Expr>::type> {};

template <typename Expr>
struct scalar_type<MatrixExpression<Expr>> : public detail::set_type<typename scalar_type<Expr>::type> {};

template <typename E1, typename E2>
struct scalar_type<MatrixAddition<E1, E2>>
    : public detail::set_type<operation_result_t<std::plus<>, typename scalar_type<E1>::type, typename scalar_type<E2>::type>> {};

template <typename E1, typename E2>
struct scalar_type<MatrixSubtraction<E1, E2>>
    : public detail::set_type<operation_result_t<std::minus<>, typename scalar_type<E1>::type, typename scalar_type<E2>::type>> {};

template <typename E1, typename E2>
struct scalar_type<ElementMatrixMultiplication<E1, E2>>
    : public detail::set_type<operation_result_t<std::multiplies<>, typename scalar_type<E1>::type, typename scalar_type<E2>::type>> {};

template <typename E1, typename E2>
struct scalar_type<MatrixMultiplication<E1, E2>> {
    using type = operation_result_t<std::multiplies<>, typename scalar_type<E1>::type, typename scalar_type<E2>::type>;
    static_assert(std::is_same<operation_result_t<std::plus<>, type, type>, type>::value,
                  "Resulting type of matrix multiplication must yield the same type if added up.");
};

template <typename T>
struct scalar_type<EvaluatedExpression<T>> : public detail::set_type<T> {};

template <typename E>
struct scalar_type<MatrixNegation<E>> : public detail::set_type<typename scalar_type<E>::type> {};

template <typename E>
struct scalar_type<MatrixTranspose<E>> : public detail::set_type<typename scalar_type<E>::type> {};

template <typename E>
struct scalar_type<MatrixConjugate<E>> : public detail::set_type<typename scalar_type<E>::type> {};

template <typename E>
struct scalar_type<MatrixAbs<E>> : public detail::set_type<typename scalar_type<E>::type> {};

template <typename E, typename U>
struct scalar_type<MatrixScalarMultiplication<E, U>> : public detail::set_type<operation_result_t<std::multiplies<>, typename scalar_type<E>::type, U>> {};

template <typename E, typename U>
struct scalar_type<ScalarMatrixMultiplication<E, U>> : public detail::set_type<operation_result_t<std::multiplies<>, U, typename scalar_type<E>::type>> {};

template <typename T>
struct scalar_type<Matrix<T>> : public detail::set_type<T> {};

template <typename T>
struct scalar_type<IdentityMatrix<T>> : public detail::set_type<T> {};

template <typename T>
struct scalar_type<PermutationMatrix<T>> : public detail::set_type<T> {};

template <typename E>
struct scalar_type<SubMatrix<E>> : public detail::set_type<typename scalar_type<E>::type> {};

template <typename T, bool C>
struct scalar_type<RefSubMatrix<T, C>> : public detail::set_type<T> {};

template <typename Expr>
using scalar_type_t = typename scalar_type<Expr>::type;

template <typename Expr>
struct vectorizable : public std::false_type {};

template <typename Expr>
struct vectorizable<const Expr> : vectorizable<Expr> {};

template <typename Expr>
struct vectorizable<volatile Expr> : vectorizable<Expr> {};

template <typename Expr>
struct vectorizable<const volatile Expr> : vectorizable<Expr> {};

template <typename E>
struct vectorizable<MatrixExpression<E>> : public vectorizable<E> {};

template <typename E1, typename E2>
struct vectorizable<MatrixAddition<E1, E2>>
    : public detail::and_value<vectorizable<E1>::value, vectorizable<E2>::value, std::is_arithmetic<scalar_type_t<MatrixAddition<E1, E2>>>::value> {};

template <typename E1, typename E2>
struct vectorizable<MatrixSubtraction<E1, E2>>
    : public detail::and_value<vectorizable<E1>::value, vectorizable<E2>::value, std::is_arithmetic<scalar_type_t<MatrixSubtraction<E1, E2>>>::value> {};

template <typename E1, typename E2>
struct vectorizable<ElementMatrixMultiplication<E1, E2>>
    : public detail::and_value<vectorizable<E1>::value, vectorizable<E2>::value,
                               std::is_arithmetic<scalar_type_t<ElementMatrixMultiplication<E1, E2>>>::value> {};

template <typename T>
struct vectorizable<EvaluatedExpression<T>> : public std::true_type {};

template <typename E>
struct vectorizable<MatrixNegation<E>> : public vectorizable<E> {};

template <typename E>
struct vectorizable<MatrixTranspose<E>> : public std::false_type {};

template <typename E>
struct vectorizable<MatrixConjugate<E>> : public std::false_type {};

template <typename E>
struct vectorizable<MatrixAbs<E>> : public vectorizable<E> {};

template <typename E, typename U>
struct vectorizable<MatrixScalarMultiplication<E, U>> : public detail::and_value<vectorizable<E>::value, std::is_same<scalar_type_t<E>, U>::value> {};

template <typename T>
struct vectorizable<Matrix<T>> : public std::is_arithmetic<T> {};

template <typename E1, typename E2>
struct vectorizable<MatrixMultiplication<E1, E2>> : public std::is_arithmetic<scalar_type_t<MatrixMultiplication<E1, E2>>> {};

template <typename E>
struct vectorizable<PermutationMatrix<E>> : public std::false_type {};

template <typename E>
struct vectorizable<SubMatrix<E>> : public std::false_type {};

template <typename T, bool C>
struct vectorizable<RefSubMatrix<T, C>> : public std::integral_constant<bool, C> {};


template <typename T>
struct vectorizable<IdentityMatrix<T>> : public std::false_type {};

template <typename Expr>
constexpr bool vectorizable_v = vectorizable<Expr>::value;

template <typename E>
struct expression_member : public detail::set_type<const E> {};

template <typename T>
struct expression_member<Matrix<T>> : public detail::set_type<const Matrix<T>&> {};

template <typename E>
using expression_member_t = typename expression_member<E>::type;

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

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
