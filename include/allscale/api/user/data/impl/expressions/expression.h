#pragma once

#include "allscale/api/user/algorithm/pfor.h"
#include "allscale/api/user/algorithm/preduce.h"
#include "allscale/api/user/data/impl/forward.h"
#include "allscale/api/user/data/impl/iterator.h"
#include "allscale/api/user/data/impl/iterator_wrapper.h"
#include "allscale/api/user/data/impl/traits.h"
#include "allscale/api/user/data/impl/types.h"
#include "allscale/utils/vector.h"

#include <Eigen/Eigen>
#include <Vc/Vc>
#include <complex>
#include <type_traits>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

namespace detail {

template <typename T1, typename T2>
std::enable_if_t<vectorizable_v<Matrix<T2>>> set_value(const T1& value, Matrix<T2>& dst) {
    using PacketScalar = typename Vc::Vector<T2>;


    const int total_size = dst.rows() * dst.columns();
    const int packet_size = PacketScalar::size();
    const int aligned_end = total_size / packet_size * packet_size;

    PacketScalar z(static_cast<T2>(value));

    algorithm::pfor(utils::Vector<coordinate_type, 1>(0), utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
        int i = coord[0] * packet_size;
        point_type p{i / dst.columns(), i % dst.columns()};

        z.store(&dst[p]);
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
void set_value(const T1& value, SubMatrix<Matrix<T2>>& dst) {
    using PacketScalar = typename Vc::Vector<T2>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = dst.columns() / packet_size * packet_size;

    PacketScalar z(static_cast<T2>(value));

    algorithm::pfor(point_type{dst.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        z.store(&dst[p]);
    });

    for(int i = 0; i < dst.rows(); ++i) {
        for(int j = caligned_end; j < dst.columns(); ++j) {
            point_type p{i, j};
            dst[p] = static_cast<T2>(value);
        }
    }
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
auto sub(const MatrixExpression<E>& e, const BlockRange& br) {
    return SubMatrix<expression_tree_t<const E>>(e, br);
}

template <typename T>
auto sub(Matrix<T>& e, const BlockRange& br) {
    return SubMatrix<Matrix<T>>(e, br);
}

template <typename E>
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
    return sub(static_cast<const E&>(e), {{r, 0}, {1, e.columns()}});
}

template <typename T>
auto row(SubMatrix<Matrix<T>> e, coordinate_type r) {
    assert_lt(r, e.rows());
    return sub(e, {{r, 0}, {1, e.columns()}});
}

template <typename T>
auto row(Matrix<T>& e, coordinate_type r) {
    assert_lt(r, e.rows());
    return sub(e, {{r, 0}, {1, e.columns()}});
}

template <typename T>
auto row(const Matrix<T>& e, coordinate_type r) {
    assert_lt(r, e.rows());
    return sub(e, {{r, 0}, {1, e.columns()}});
}

template <typename E>
auto column(const MatrixExpression<E>& e, coordinate_type c) {
    assert_lt(c, e.columns());
    return sub(static_cast<const E&>(e), {{0, c}, {e.rows(), 1}});
}

template <typename T>
auto column(SubMatrix<Matrix<T>> e, coordinate_type c) {
    assert_lt(c, e.columns());
    return sub(e, {{0, c}, {e.rows(), 1}});
}

template <typename T>
auto column(Matrix<T>& e, coordinate_type c) {
    assert_lt(c, e.columns());
    return sub(e, {{0, c}, {e.rows(), 1}});
}

template <typename T>
auto column(const Matrix<T>& e, coordinate_type c) {
    assert_lt(c, e.columns());
    return sub(e, {{0, c}, {e.rows(), 1}});
}

} // end namespace detail

template <typename E>
class MatrixExpression {
    static_assert(std::is_same<E, std::remove_reference_t<E>>::value, "A MatrixExpression type may not be a reference type.");

public:
    using T = scalar_type_t<E>;
    using PacketScalar = typename Vc::Vector<T>;

private:
    E& impl() {
        return static_cast<E&>(*this);
    }

    const E& impl() const {
        return static_cast<const E&>(*this);
    }

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
        return detail::sub(impl(), {{p.x, 0}, {p.y, columns()}});
    }

    auto rowRange(range_type p) const {
        return detail::sub(impl(), {{p.x, 0}, {p.y, columns()}});
    }

    auto columnRange(range_type p) {
        return detail::sub(impl(), {{0, p.x}, {rows(), p.y}});
    }

    auto columnRange(range_type p) const {
        return detail::sub(impl(), {{0, p.x}, {rows(), p.y}});
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
    ElementMatrixMultiplication<expression_tree_t<const E>, expression_tree_t<const E2>> product(const MatrixExpression<E2>& e) const {
        return ElementMatrixMultiplication<expression_tree_t<const E>, expression_tree_t<const E2>>(impl(), e);
    }

    MatrixTranspose<expression_tree_t<const E>> transpose() const;

    MatrixConjugate<expression_tree_t<const E>> conjugate() const {
        return MatrixConjugate<expression_tree_t<const E>>(impl());
    }

    MatrixTranspose<MatrixConjugate<const E>> adjoint() const {
        return this->conjugate().transpose();
    }

    auto sub(BlockRange block_range) {
        return detail::sub(impl(), block_range);
    }

    auto sub(BlockRange block_range) const {
        return detail::sub(impl(), block_range);
    }

    MatrixAbs<expression_tree_t<const E>> abs() const {
        return MatrixAbs<expression_tree_t<const E>>(impl());
    }

    template <ViewType View>
    MatrixView<expression_tree_t<const E>, View> view() const {
        return MatrixView<expression_tree_t<const E>, View>(impl());
    }

    T norm() const {
        return std::sqrt(product(*this).accumulate());
    }

    // -- defined in decomposition.h
    LUD<T> LUDecomposition() const;
    FPLUD<T> FPLUDecomposition() const;
    QRD<T> QRDecomposition() const;
    SVD<T> SVDecomposition() const;

    // Avoid that temporaries bind to the const& ref-qualified member function
    Iterator<E> begin() const&& = delete;

    Iterator<E> begin() const& {
        return cbegin();
    }

    // Avoid that temporaries bind to the const& ref-qualified member function
    Iterator<E> end() const&& = delete;

    Iterator<E> end() const& {
        return cend();
    }

    // Avoid that temporaries bind to the const& ref-qualified member function
    Iterator<E> cbegin() const&& = delete;

    Iterator<E> cbegin() const& {
        return Iterator<E>(*this, 0);
    }

    // Avoid that temporaries bind to the const& ref-qualified member function
    Iterator<E> cend() const&& = delete;

    Iterator<E> cend() const& {
        return Iterator<E>(*this, rows() * columns());
    }

    T accumulate() const {
        return std::accumulate(begin(), end(), static_cast<T>(0));
    }

    template <typename Reducer>
    T reduce(T init, Reducer f) const {
        return algorithm::preduce(begin(), end(), [&](const T& a, T& b) { b = f(a, b); }, [&](const T& a, const T& b) { return f(a, b); },
                                  [&]() { return init; })
            .get();
    }

    template <typename Reducer>
    T reduce(Reducer f) const {
        assert_gt(rows() * columns(), 0);
        return algorithm::preduce(begin() + 1, end(), [&](const T& a, T& b) { b = f(a, b); }, [&](const T& a, const T& b) { return f(a, b); },
                                  [&]() { return *begin(); })
            .get();
    }

    template <typename Reducer>
    Iterator<E> iterator_reduce(Reducer f) const {
        assert_gt(rows() * columns(), 0);
        return algorithm::preduce(IteratorWrapper<Iterator<E>>(begin() + 1), IteratorWrapper<Iterator<E>>(end()),
                                  [&](const Iterator<E>& a, Iterator<E>& b) { b = f(a, b); }, f, [&]() { return begin(); })
            .get();
    }

    T max() const {
        return reduce([](const T& a, const T& b) { return std::max(a, b); });
    }

    T min() const {
        return reduce([](const T& a, const T& b) { return std::min(a, b); });
    }

    // Avoid that temporaries bind to the const& ref-qualified member function
    Iterator<E> max_element() const&& = delete;

    Iterator<E> max_element() const& {
        Iterator<E> max = begin();

        for(auto it = begin(); it != end(); ++it) {
            if(*max < *it)
                max = it;
        }

        return max;
    }

    // Avoid that temporaries bind to the const& ref-qualified member function
    Iterator<E> min_element() const&& = delete;

    Iterator<E> min_element() const& {
        return iterator_reduce([](const Iterator<E>& a, const Iterator<E>& b) { return (*b < *a) ? b : a; });
    }

    // -- defined in eigen.h
    EigenSolver<T> solveEigen() const;

    // -- defined in decomposition.h
    T determinant() const;
    Matrix<T> inverse() const;

    template <typename simd_type = PacketScalar>
    std::enable_if_t<vectorizable_v<E>, simd_type> packet(point_type p) const {
        return impl().template packet<simd_type>(p);
    }

    // -- defined in evaluate.h
    auto eval() -> eval_return_t<std::remove_reference_t<decltype(impl())>>;
    auto eval() const -> eval_return_t<std::remove_reference_t<decltype(impl())>>;

    operator E&() {
        return impl();
    }

    operator const E&() const {
        return impl();
    }
};

} // namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
