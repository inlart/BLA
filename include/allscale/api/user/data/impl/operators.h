#include "expressions.h"
#include "forward.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <typename T, typename E>
Matrix<T>& operator+=(Matrix<T>& u, const MatrixExpression<E>& v) {
	// TODO: handle aliasing
	evaluate(MatrixAddition<Matrix<T>, E>(u, v), u);
	return u;
}

template <typename T, typename E>
Matrix<T>& operator-=(Matrix<T>& u, const MatrixExpression<E>& v) {
	// TODO: handle aliasing
	evaluate(MatrixSubtraction<Matrix<T>, E>(u, v), u);
	return u;
}

template <typename T, typename E>
Matrix<T>& operator*=(Matrix<T>& u, const MatrixExpression<E>& v) {
	assert(v.columns() == v.rows());
	// no aliasing because the result is written in a temporary matrix
	Matrix<T> tmp(u * v);
	u = tmp;
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
	if(a.size() != b.size()) return false;

	for(coordinate_type i = 0; i < a.rows(); ++i) {
		for(coordinate_type j = 0; j < a.columns(); ++j) {
			if(a[{i, j}] != b[{i, j}]) return false;
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
std::ostream& operator<<(std::ostream& os, MatrixExpression<E> const& m) {
	for(coordinate_type i = 0; i < m.rows(); ++i) {
		for(coordinate_type j = 0; j < m.columns(); ++j) {
			os << m[{i, j}] << " ";
		}
		if(i + 1 < m.rows()) { os << std::endl; }
	}
	return os;
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
