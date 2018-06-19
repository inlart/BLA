#include <type_traits>

#include "expressions.h"
#include "traits.h"
#include "types.h"


namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {


namespace detail {

// -- evaluate a matrix expression using vectorization
template <typename E>
std::enable_if_t<vectorizable_v<E>> evaluate(const MatrixExpression<E>& expression, Matrix<scalar_type_t<E>>& dst) {
    expression_member_t<decltype(simplify(expression))> expr = simplify(expression);

    using T = scalar_type_t<E>;
    using PacketScalar = typename Vc::native_simd<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = expr.columns() / packet_size * packet_size;

    algorithm::pfor(point_type{expr.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        expr.template packet<PacketScalar, Vc::flags::element_aligned_tag>(p).copy_to(&dst[p], Vc::flags::element_aligned);
    });

    for(int i = 0; i < expr.rows(); ++i) {
        for(int j = caligned_end; j < expr.columns(); ++j) {
            dst[{i, j}] = expr[{i, j}];
        }
    }
}

// -- evaluate a matrix expression by simply copying each value
template <typename E>
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expression, Matrix<scalar_type_t<E>>& dst) {
    expression_member_t<decltype(simplify(expression))> expr = simplify(expression);

    algorithm::pfor(expr.size(), [&](const auto& pos) { dst[pos] = expr[pos]; });
}

template <typename E, bool C>
std::enable_if_t<vectorizable_v<E>> evaluate(const MatrixExpression<E>& expression, RefSubMatrix<scalar_type_t<E>, C>& dst) {
    expression_member_t<decltype(simplify(expression))> expr = simplify(expression);

    using T = scalar_type_t<E>;
    using PacketScalar = typename Vc::native_simd<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = expr.columns() / packet_size * packet_size;

    algorithm::pfor(point_type{expr.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        expr.template packet<PacketScalar, Vc::flags::element_aligned_tag>(p).copy_to(&dst[p], Vc::flags::element_aligned);
    });

    for(int i = 0; i < expr.rows(); ++i) {
        for(int j = caligned_end; j < expr.columns(); ++j) {
            dst[{i, j}] = expr[{i, j}];
        }
    }
}

template <typename E, bool C>
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expression, RefSubMatrix<scalar_type_t<E>, C>& dst) {
    expression_member_t<decltype(simplify(expression))> expr = simplify(expression);

    algorithm::pfor(expr.size(), [&](const auto& pos) { dst[pos] = expr[pos]; });
}

template <typename E>
auto eval(const MatrixExpression<E>& e) -> Matrix<scalar_type_t<E>> {
    using T = scalar_type_t<E>;
    Matrix<T> tmp(e.size());

    detail::evaluate(e, tmp);

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
    detail::evaluate(mat, *this);
}

template <typename T, bool C>
template <typename E>
void RefSubMatrix<T, C>::evaluate(const MatrixExpression<E>& mat) {
    detail::evaluate(mat, *this);
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
