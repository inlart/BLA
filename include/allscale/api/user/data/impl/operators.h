#pragma once

#include <allscale/api/user/data/grid.h>
#include <allscale/utils/assert.h>
#include <iostream>

#include "allscale/api/user/data/impl/expressions.h"
#include "allscale/api/user/data/impl/types.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename T, typename E>
Matrix<T>& operator+=(Matrix<T>& u, const MatrixExpression<E>& v) {
    // TODO: handle aliasing
    detail::evaluate(MatrixAddition<Matrix<T>, E>(u, v), &u[{0, 0}]);
    return u;
}

template <typename T, typename E>
RefSubMatrix<T, true> operator+=(RefSubMatrix<T, true> u, const MatrixExpression<E>& v) {
    // TODO: handle aliasing
    detail::evaluate(MatrixAddition<RefSubMatrix<T, true>, E>(u, v), &u[{0, 0}]);
    return u;
}

template <typename T, typename E>
RefSubMatrix<T, false> operator+=(RefSubMatrix<T, false> u, const MatrixExpression<E>& v) {
    // TODO: handle aliasing
    algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] += v[pos]; });
    return u;
}

template <typename T, typename E>
Matrix<T>& operator-=(Matrix<T>& u, const MatrixExpression<E>& v) {
    // TODO: handle aliasing
    detail::evaluate(MatrixSubtraction<Matrix<T>, E>(u, v), &u[{0, 0}]);
    return u;
}

template <typename T, typename E>
RefSubMatrix<T, true> operator-=(RefSubMatrix<T, true> u, const MatrixExpression<E>& v) {
    // TODO: handle aliasing
    detail::evaluate(MatrixSubtraction<RefSubMatrix<T, true>, E>(u, v), &u[{0, 0}]);
    return u;
}

template <typename T, typename E>
RefSubMatrix<T, false> operator-=(RefSubMatrix<T, false> u, const MatrixExpression<E>& v) {
    // TODO: handle aliasing
    algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] -= v[pos]; });
    return u;
}

template <typename T, typename E>
Matrix<T>& operator*=(Matrix<T>& u, const MatrixExpression<E>& v) {
    assert_eq(v.columns(), v.rows());
    // no aliasing because the result is written in a temporary matrix
    Matrix<T> tmp(u * v);
    u = tmp;
    return u;
}

template <typename T>
std::enable_if_t<vectorizable_v<Matrix<T>>, Matrix<T>&> operator*=(Matrix<T>& u, const T& v) {
    // no aliasing because the result is written in a temporary matrix
    using PacketScalar = typename Vc::native_simd<T>;


    const int total_size = u.rows() * u.columns();
    const int packet_size = PacketScalar::size();
    const int aligned_end = total_size / packet_size * packet_size;

    const PacketScalar simd_value(v);

    algorithm::pfor(utils::Vector<coordinate_type, 1>(0), utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
        int i = coord[0] * packet_size;
        point_type p{i / u.columns(), i % u.columns()};
        (u.template packet<PacketScalar, detail::alignment_t<PacketScalar>>(p) * simd_value).copy_to(&u[p], detail::alignment_t<PacketScalar>{});
    });

    for(int i = aligned_end; i < total_size; i++) {
        point_type p{i / u.columns(), i % u.columns()};
        u[p] *= v;
    }
    return u;
}

template <typename T>
std::enable_if_t<!vectorizable_v<Matrix<T>>, Matrix<T>&> operator*=(Matrix<T>& u, const T& v) {
    // no aliasing because the result is written in a temporary matrix
    algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] *= v; });

    return u;
}

template <typename T, bool C>
std::enable_if_t<vectorizable_v<RefSubMatrix<T, C>>, RefSubMatrix<T, C>> operator*=(RefSubMatrix<T, C> u, const T& v) {
    using PacketScalar = typename Vc::native_simd<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = u.columns() / packet_size * packet_size;

    const PacketScalar simd_value(v);

    algorithm::pfor(point_type{u.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        (u.template packet<PacketScalar, Vc::flags::element_aligned_tag>(p) * simd_value).copy_to(&u[p], Vc::flags::element_aligned);
    });

    for(int i = 0; i < u.rows(); ++i) {
        for(int j = caligned_end; j < u.columns(); ++j) {
            u[{i, j}] *= v;
        }
    }


    return u;
}

template <typename T, bool C>
std::enable_if_t<!vectorizable_v<RefSubMatrix<T, C>>, RefSubMatrix<T, C>> operator*=(RefSubMatrix<T, C> u, const T& v) {
    // no aliasing because the result is written in a temporary matrix
    algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] *= v; });

    return u;
}

template <typename T>
std::enable_if_t<vectorizable_v<Matrix<T>>, Matrix<T>&> operator/=(Matrix<T>& u, const T& v) {
    // no aliasing because the result is written in a temporary matrix
    using PacketScalar = typename Vc::native_simd<T>;


    const int total_size = u.rows() * u.columns();
    const int packet_size = PacketScalar::size();
    const int aligned_end = total_size / packet_size * packet_size;

    const PacketScalar simd_value(v);

    algorithm::pfor(utils::Vector<coordinate_type, 1>(0), utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
        int i = coord[0] * packet_size;
        point_type p{i / u.columns(), i % u.columns()};
        (u.template packet<PacketScalar, detail::alignment_t<PacketScalar>>(p) / simd_value).copy_to(&u[p], detail::alignment_t<PacketScalar>{});
    });

    for(int i = aligned_end; i < total_size; i++) {
        point_type p{i / u.columns(), i % u.columns()};
        u[p] /= v;
    }
    return u;
}

template <typename T>
std::enable_if_t<!vectorizable_v<Matrix<T>>, Matrix<T>&> operator/=(Matrix<T>& u, const T& v) {
    // no aliasing because the result is written in a temporary matrix
    algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] /= v; });

    return u;
}

template <typename T, bool C>
std::enable_if_t<vectorizable_v<RefSubMatrix<T, C>>, RefSubMatrix<T, C>> operator/=(RefSubMatrix<T, C> u, const T& v) {
    using PacketScalar = typename Vc::native_simd<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = u.columns() / packet_size * packet_size;

    const PacketScalar simd_value(v);

    algorithm::pfor(point_type{u.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        (u.template packet<PacketScalar, Vc::flags::element_aligned_tag>(p) / simd_value).copy_to(&u[p], Vc::flags::element_aligned);
    });

    for(int i = 0; i < u.rows(); ++i) {
        for(int j = caligned_end; j < u.columns(); ++j) {
            u[{i, j}] /= v;
        }
    }


    return u;
}

template <typename T, bool C>
std::enable_if_t<!vectorizable_v<RefSubMatrix<T, C>>, RefSubMatrix<T, C>> operator/=(RefSubMatrix<T, C> u, const T& v) {
    // no aliasing because the result is written in a temporary matrix

    algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] /= v; });

    return u;
}


// -- matrix matrix addition
template <typename E1, typename E2>
MatrixAddition<E1, E2> const operator+(const MatrixExpression<E1>& u, const MatrixExpression<E2>& v) {
    return MatrixAddition<E1, E2>(u, v);
}

// -- matrix matrix subtraction
template <typename E1, typename E2>
MatrixSubtraction<E1, E2> const operator-(const MatrixExpression<E1>& u, const MatrixExpression<E2>& v) {
    return MatrixSubtraction<E1, E2>(u, v);
}

// -- matrix negation
template <typename E>
MatrixNegation<E> const operator-(const MatrixExpression<E>& e) {
    return MatrixNegation<E>(e);
}

// -- scalar * matrix multiplication
// Note: without the std::enable_if a matrix * matrix multiplication would be ambiguous
template <typename E, typename U>
std::enable_if_t<!std::is_base_of<MatrixExpression<U>, U>::value, ScalarMatrixMultiplication<E, U>> operator*(const U& u, const MatrixExpression<E>& v) {
    return ScalarMatrixMultiplication<E, U>(u, v);
}

template <typename E, typename U>
std::enable_if_t<!std::is_base_of<MatrixExpression<U>, U>::value, MatrixScalarMultiplication<E, U>> operator*(const MatrixExpression<E>& v, const U& u) {
    return MatrixScalarMultiplication<E, U>(v, u);
}

// -- matrix * matrix multiplication
template <typename E1, typename E2>
MatrixMultiplication<E1, E2> operator*(const MatrixExpression<E1>& u, const MatrixExpression<E2>& v) {
    return MatrixMultiplication<E1, E2>(u, v);
}

template <typename E1, typename E2>
bool operator==(const MatrixExpression<E1>& a, const MatrixExpression<E2>& b) {
    if(a.size() != b.size())
        return false;

    for(coordinate_type i = 0; i < a.rows(); ++i) {
        for(coordinate_type j = 0; j < a.columns(); ++j) {
            if(a[{i, j}] != b[{i, j}])
                return false;
        }
    }

    return true;
}

template <typename E1, typename E2>
bool operator!=(const MatrixExpression<E1>& a, const MatrixExpression<E2>& b) {
    return !(a == b);
}

// -- print a matrix expression
template <typename E>
std::ostream& operator<<(std::ostream& os, const MatrixExpression<E>& m) {
    for(coordinate_type i = 0; i < m.rows(); ++i) {
        for(coordinate_type j = 0; j < m.columns(); ++j) {
            os << m[{i, j}] << " ";
        }
        if(i + 1 < m.rows()) {
            os << std::endl;
        }
    }
    return os;
}

template <typename T>
class MatrixInitializer {
public:
    template <typename T2>
    MatrixInitializer(Matrix<T>& m, T2&& val) : matrix(m), pos({0, 0}) {
        *this, std::move(val);
    }

    ~MatrixInitializer() {
        assert_eq(pos, (point_type{matrix.rows(), 0}));
    }

    template <typename T2>
    MatrixInitializer& operator,(T2&& val) {
        assert_lt(pos, matrix.size());
        matrix[pos] = static_cast<T>(val);

        increasePos();

        return *this;
    }

private:
    void increasePos() {
        pos.y++;
        if(pos.y == matrix.columns()) {
            pos.y = 0;
            pos.x++;
        }
    }

private:
    Matrix<T>& matrix;
    point_type pos;
};

template <typename T1, typename T2>
MatrixInitializer<T1> operator<<(Matrix<T1>& m, T2&& val) {
    return MatrixInitializer<T1>(m, std::move(val));
}


} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
