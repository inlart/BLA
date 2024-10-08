#pragma once

#include <allscale/api/user/data/grid.h>
#include <allscale/utils/assert.h>
#include <iostream>

#include "bla/impl/expressions.h"
#include "bla/impl/simplify.h"
#include "bla/impl/types.h"

namespace bla {
namespace impl {

// -- in-place Matrix addition
template <typename T, typename E>
Matrix<T>& operator+=(Matrix<T>& u, const MatrixExpression<E>& v) {
    detail::evaluate_simplify(u + v, u);
    return u;
}

// -- in-place SubMatrix addition
template <typename T, typename E>
SubMatrix<Matrix<T>> operator+=(SubMatrix<Matrix<T>> u, const MatrixExpression<E>& v) {
    detail::evaluate_simplify(u + v, u);
    return u;
}

// -- in-place Matrix subtraction
template <typename T, typename E>
Matrix<T>& operator-=(Matrix<T>& u, const MatrixExpression<E>& v) {
    detail::evaluate_simplify(u - v, u);

    return u;
}

// -- in-place SubMatrix subtraction
template <typename T, typename E>
SubMatrix<Matrix<T>> operator-=(SubMatrix<Matrix<T>> u, const MatrixExpression<E>& v) {
    detail::evaluate_simplify(u - v, u);

    return u;
}

// -- in-place Matrix multiplication
template <typename T, typename E>
Matrix<T>& operator*=(Matrix<T>& u, const MatrixExpression<E>& v) {
    assert_eq(v.columns(), v.rows());
    // no aliasing because the result is written in a temporary matrix
    Matrix<T> tmp(u * v);
    u = tmp;
    return u;
}

// -- multiply Matrix by value v using vectorization
template <typename T>
std::enable_if_t<vectorizable_v<Matrix<T>>, Matrix<T>&> operator*=(Matrix<T>& u, const T& v) {
    using PacketScalar = typename Vc::Vector<T>;


    const int total_size = u.rows() * u.columns();
    const int packet_size = PacketScalar::size();
    const int aligned_end = total_size / packet_size * packet_size;

    const PacketScalar simd_value(v);

    // multiply value v with each value of the matrix using vectorization
    allscale::api::user::algorithm::pfor(allscale::utils::Vector<coordinate_type, 1>(0), allscale::utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
        int i = coord[0] * packet_size;
        point_type p{i / u.columns(), i % u.columns()};
        (u.template packet<point_type, PacketScalar>(p) * simd_value).store(&u[p]);
    });

    // calculate what can't be vectorized
    for(int i = aligned_end; i < total_size; i++) {
        point_type p{i / u.columns(), i % u.columns()};
        u[p] *= v;
    }
    return u;
}

// -- multiply Matrix by value v
template <typename T>
std::enable_if_t<!vectorizable_v<Matrix<T>>, Matrix<T>&> operator*=(Matrix<T>& u, const T& v) {
    allscale::api::user::algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] *= v; });

    return u;
}

// -- multiply SubMatrix by value v using vectorization
template <typename T>
std::enable_if_t<vectorizable_v<SubMatrix<Matrix<T>>>, SubMatrix<Matrix<T>>> operator*=(SubMatrix<Matrix<T>> u, const T& v) {
    using PacketScalar = typename Vc::Vector<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = u.columns() / packet_size * packet_size;

    const PacketScalar simd_value(v);

    // multiply value v with each value of the matrix using vectorization
    allscale::api::user::algorithm::pfor(point_type{u.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        (u.template packet<point_type, PacketScalar>(p) * simd_value).store(&u[p]);
    });

    // calculate what can't be vectorized
    for(int i = 0; i < u.rows(); ++i) {
        for(int j = caligned_end; j < u.columns(); ++j) {
            u[{i, j}] *= v;
        }
    }


    return u;
}

// -- multiply SubMatrix by value v
template <typename T, bool V>
std::enable_if_t<!vectorizable_v<SubMatrix<Matrix<T>>>, SubMatrix<Matrix<T>>> operator*=(SubMatrix<Matrix<T>> u, const T& v) {
    allscale::api::user::algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] *= v; });

    return u;
}

// -- divide Matrix by value v using vectorization
template <typename T>
std::enable_if_t<vectorizable_v<Matrix<T>>, Matrix<T>&> operator/=(Matrix<T>& u, const T& v) {
    using PacketScalar = typename Vc::Vector<T>;


    const int total_size = u.rows() * u.columns();
    const int packet_size = PacketScalar::size();
    const int aligned_end = total_size / packet_size * packet_size;

    const PacketScalar simd_value(v);

    // divide each value of the matrix by v using vectorization
    allscale::api::user::algorithm::pfor(allscale::utils::Vector<coordinate_type, 1>(0), allscale::utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
        int i = coord[0] * packet_size;
        point_type p{i / u.columns(), i % u.columns()};
        (u.template packet<point_type, PacketScalar>(p) / simd_value).store(&u[p]);
    });

    // calculate what can't be vectorized
    for(int i = aligned_end; i < total_size; i++) {
        point_type p{i / u.columns(), i % u.columns()};
        u[p] /= v;
    }
    return u;
}

// -- multiply Matrix by value v
template <typename T>
std::enable_if_t<!vectorizable_v<Matrix<T>>, Matrix<T>&> operator/=(Matrix<T>& u, const T& v) {
    allscale::api::user::algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] /= v; });

    return u;
}

// -- divide SubMatrix by value v using vectorization
template <typename T>
std::enable_if_t<vectorizable_v<SubMatrix<Matrix<T>>>, SubMatrix<Matrix<T>>> operator/=(SubMatrix<Matrix<T>> u, const T& v) {
    using PacketScalar = typename Vc::Vector<T>;

    const int packet_size = PacketScalar::size();
    const int caligned_end = u.columns() / packet_size * packet_size;

    const PacketScalar simd_value(v);

    // divide each value of the matrix by v using vectorization
    allscale::api::user::algorithm::pfor(point_type{u.rows(), caligned_end / packet_size}, [&](const auto& coord) {
        int j = coord.y * packet_size;
        point_type p{coord.x, j};
        (u.template packet<point_type, PacketScalar>(p) / simd_value).store(&u[p]);
    });

    // calculate what can't be vectorized
    for(int i = 0; i < u.rows(); ++i) {
        for(int j = caligned_end; j < u.columns(); ++j) {
            u[{i, j}] /= v;
        }
    }


    return u;
}

// -- divide SubMatrix by value v
template <typename T>
std::enable_if_t<!vectorizable_v<SubMatrix<Matrix<T>>>, SubMatrix<Matrix<T>>> operator/=(SubMatrix<Matrix<T>> u, const T& v) {
    allscale::api::user::algorithm::pfor(u.size(), [&](const auto& pos) { u[pos] /= v; });

    return u;
}


// -- matrix matrix addition
template <typename E1, typename E2>
MatrixAddition<expression_tree_t<const E1>, expression_tree_t<const E2>> const operator+(const MatrixExpression<E1>& u, const MatrixExpression<E2>& v) {
    return MatrixAddition<expression_tree_t<const E1>, expression_tree_t<const E2>>(u, v);
}

// -- matrix matrix subtraction
template <typename E1, typename E2>
MatrixSubtraction<expression_tree_t<const E1>, expression_tree_t<const E2>> const operator-(const MatrixExpression<E1>& u, const MatrixExpression<E2>& v) {
    return MatrixSubtraction<expression_tree_t<const E1>, expression_tree_t<const E2>>(u, v);
}

// -- matrix negation
template <typename E>
MatrixNegation<expression_tree_t<const E>> const operator-(const MatrixExpression<E>& e) {
    return MatrixNegation<expression_tree_t<const E>>(e);
}

// -- scalar * matrix multiplication
// Note: without the std::enable_if a matrix * matrix multiplication would be ambiguous
template <typename E, typename U>
std::enable_if_t<!std::is_base_of<MatrixExpression<U>, U>::value, ScalarMatrixMultiplication<expression_tree_t<const E>, U>>
operator*(const U& u, const MatrixExpression<E>& v) {
    return ScalarMatrixMultiplication<expression_tree_t<const E>, U>(u, v);
}

// -- matrix * scalar multiplication
template <typename E, typename U>
std::enable_if_t<!std::is_base_of<MatrixExpression<U>, U>::value, MatrixScalarMultiplication<expression_tree_t<const E>, U>>
operator*(const MatrixExpression<E>& v, const U& u) {
    return MatrixScalarMultiplication<expression_tree_t<const E>, U>(v, u);
}

// -- matrix * matrix multiplication
template <typename E1, typename E2>
auto operator*(const MatrixExpression<E1>& u, const MatrixExpression<E2>& v) {
    return MatrixMultiplication<expression_tree_t<const E1>, expression_tree_t<const E2>>(u, v);
}

// -- matrix equality check
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

// -- matrix inequality check
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

// -- class to initialize a matrix
template <typename T>
class MatrixInitializer {
public:
    template <typename T2>
    MatrixInitializer(Matrix<T>& m, T2&& val) : matrix(m), pos({0, 0}) {
        *this, std::move(val);
    }

    ~MatrixInitializer() {
        // check if the matrix is fully filled
        assert_eq(pos, (point_type{matrix.rows(), 0}));
    }

    template <typename T2>
    MatrixInitializer& operator,(T2&& val) {
        assert_lt(pos, matrix.size());
        // add value to matrix
        matrix[pos] = static_cast<T>(val);

        // increase position for the next element
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
} // namespace bla
