
namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {


namespace detail {

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
        expr.template packet<PacketScalar, alignment_t<PacketScalar>>(p).copy_to(dst + i, alignment_t<PacketScalar>{});
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

} // namespace detail

template <typename E>
auto MatrixExpression<E>::eval() {
    return detail::eval(static_cast<E&>(*this));
}

template <typename E>
auto MatrixExpression<E>::eval() const {
    return detail::eval(static_cast<const E&>(*this));
}

template <typename T>
template <typename E>
void Matrix<T>::evaluate(const MatrixExpression<E>& mat) {
    detail::evaluate(mat, &(*this)[{0, 0}]);
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
