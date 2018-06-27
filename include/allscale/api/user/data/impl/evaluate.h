#pragma once

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

template <typename T, bool C>
void swap(RefSubMatrix<T, C> a, RefSubMatrix<T, C> b) {
    assert_eq(a.size(), b.size());
    using PacketScalar = typename Vc::native_simd<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = a.columns() / packet_size * packet_size;

    algorithm::pfor(point_type{a.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};

        PacketScalar tmp = a.template packet<PacketScalar, Vc::flags::element_aligned_tag>(p);
        b.template packet<PacketScalar, Vc::flags::element_aligned_tag>(p).copy_to(&a[p], Vc::flags::element_aligned);
        tmp.copy_to(&b[p], Vc::flags::element_aligned);
    });

    for(int i = 0; i < a.rows(); ++i) {
        for(int j = caligned_end; j < a.columns(); ++j) {
            std::swap(a[{i, j}], b[{i, j}]);
        }
    }
}

// -- evaluate a matrix expression using vectorization
template <typename E>
std::enable_if_t<vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, Matrix<scalar_type_t<E>>& dst) {
    assert_eq(expression.size(), dst.size());

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
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, Matrix<scalar_type_t<E>>& dst) {
    algorithm::pfor(expr.size(), [&](const auto& pos) { dst[pos] = expr[pos]; });
}

template <typename E, bool C>
std::enable_if_t<vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, RefSubMatrix<scalar_type_t<E>, C> dst) {
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
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, RefSubMatrix<scalar_type_t<E>, C> dst) {
    algorithm::pfor(expr.size(), [&](const auto& pos) { dst[pos] = expr[pos]; });
}

} // namespace detail


} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
