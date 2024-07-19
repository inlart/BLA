#pragma once

#include <type_traits>

#include "expressions.h"
#include "traits.h"
#include "types.h"


namespace bla {
namespace impl {


namespace detail {

// -- swap the content of two SubMatrix expressions
template <typename T>
void swap(SubMatrix<Matrix<T>> a, SubMatrix<Matrix<T>> b) {
    assert_eq(a.size(), b.size());
    using PacketScalar = typename Vc::Vector<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = a.columns() / packet_size * packet_size;

    // use vectorization to swap until no more vectorization possible
    allscale::api::user::algorithm::pfor(point_type{a.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};

        PacketScalar tmp = a.template packet<PacketScalar>(p);
        b.template packet<PacketScalar>(p).store(&a[p]);
        tmp.store(&b[p]);
    });

    // swap the rest of each row
    for(int i = 0; i < a.rows(); ++i) {
        for(int j = caligned_end; j < a.columns(); ++j) {
            std::swap(a[{i, j}], b[{i, j}]);
        }
    }
}

// -- evaluate a matrix expression using vectorization

template <typename E>
std::enable_if_t<vectorizable_v<E> && is_contiguous_v<E>> evaluate(const MatrixExpression<E>& expr, Matrix<scalar_type_t<E>>& dst) {
    assert_eq(expr.size(), dst.size());

    using T = scalar_type_t<E>;
    using PacketScalar = typename Vc::Vector<T>;

    constexpr coordinate_type packet_size = PacketScalar::size();
    const coordinate_type num_vectors = (expr.columns() * expr.rows()) / packet_size;

    allscale::api::user::algorithm::pfor(allscale::utils::Vector<coordinate_type, 1>(num_vectors), [&](const auto& i) {
        const auto vector_start = i[0] * packet_size;
        const point_type p{vector_start / expr.columns(), vector_start % expr.columns()};
        expr.template packet<PacketScalar>(p).store(&dst[p], Vc::Aligned | Vc::Streaming);
    });

    for(auto i = num_vectors * packet_size; i < expr.columns() * expr.rows(); ++i) {
        const point_type p{i / expr.columns(), i % expr.columns()};
        dst[p] = expr[p];
    }
}

template <typename E>
std::enable_if_t<vectorizable_v<E> && !is_contiguous_v<E>> evaluate(const MatrixExpression<E>& expr, Matrix<scalar_type_t<E>>& dst) {
    assert_eq(expr.size(), dst.size());

    using T = scalar_type_t<E>;
    using PacketScalar = typename Vc::Vector<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = expr.columns() / packet_size * packet_size;

    // use vectorization to store values until no more vectorization possible
    allscale::api::user::algorithm::pfor(point_type{expr.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        expr.template packet<PacketScalar>(p).store(&dst[p]);
    });

    // store the rest of each row
    for(int i = 0; i < expr.rows(); ++i) {
        for(int j = caligned_end; j < expr.columns(); ++j) {
            dst[{i, j}] = expr[{i, j}];
        }
    }
}

// -- evaluate a matrix expression by simply copying each value
template <typename E>
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, Matrix<scalar_type_t<E>>& dst) {
    allscale::api::user::algorithm::pfor(expr.size(), [&](const auto& pos) { dst[pos] = expr[pos]; });
}

// -- evaluate a matrix expression using vectorization
template <typename E>
std::enable_if_t<vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, SubMatrix<Matrix<scalar_type_t<E>>> dst) {
    using T = scalar_type_t<E>;
    using PacketScalar = typename Vc::Vector<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = expr.columns() / packet_size * packet_size;

    // use vectorization to store values until no more vectorization possible
    allscale::api::user::algorithm::pfor(point_type{expr.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        expr.template packet<PacketScalar>(p).store(&dst[p]);
    });

    // store the rest of each row
    for(int i = 0; i < expr.rows(); ++i) {
        for(int j = caligned_end; j < expr.columns(); ++j) {
            dst[{i, j}] = expr[{i, j}];
        }
    }
}

// -- evaluate a matrix expression by simply copying each value
template <typename E>
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, SubMatrix<Matrix<scalar_type_t<E>>> dst) {
    allscale::api::user::algorithm::pfor(expr.size(), [&](const auto& pos) { dst[pos] = expr[pos]; });
}

// -- avoid copy of evaluated expressions
template <typename T>
void evaluate(EvaluatedExpression<T>& expr, Matrix<T>& dst) {
    dst = std::move(expr).getMatrix();
}

} // namespace detail


} // end namespace impl
} // namespace bla
