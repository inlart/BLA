#pragma once

#include <Eigen/Dense>
#include <Vc/Vc>
#include <allscale/api/user/algorithm/async.h>
#include <allscale/api/user/data/grid.h>
#include <cblas.h> // BLAS
#include <cmath>
#include <cstdlib>

namespace allscale {
namespace api {
namespace user {
namespace data {

using namespace allscale::api::core;

using coordinate_type = std::int64_t;
using point_type = GridPoint<2>;
using triple_type = GridPoint<3>;

/*
 * The base class for all matrix expressions
 * Elements are not modifiable
 */
template <typename E>
class MatrixExpression;

/*
 * Represents the sum of two MatrixExpressions E1 and E2
 */
template <typename E1, typename E2>
class MatrixAddition;

/*
 * Represents the subtraction of MatrixExpressions E1 and E2
 */
template <typename E1, typename E2>
class MatrixSubtraction;

/*
 * Represents the negation of the MatrixExpression E
 */
template <typename E>
class MatrixNegation;

/*
 * Represents the transposed MatrixExpression E
 */
template <typename E>
class MatrixTranspose;

/*
 * Represents the multiplication of a scalar with a matrix
 */
template <typename E, typename U>
class MatrixScalarMultiplication;

// Helper
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

/*
 * Represents the Matrix
 * Elements are modifiable
 * Guarantees contiguous memory
 */
template <typename T = double>
class Matrix;


template <typename Expr>
struct scalar_type;

template <typename Expr>
struct scalar_type<const Expr> : public set_type<typename scalar_type<Expr>::type> {};

template <typename Expr>
struct scalar_type<volatile Expr> : public set_type<typename scalar_type<Expr>::type> {};

template <typename Expr>
struct scalar_type<const volatile Expr> : public set_type<typename scalar_type<Expr>::type> {};

template <typename Expr>
struct scalar_type<MatrixExpression<Expr>> : public set_type<typename scalar_type<Expr>::type> {};

template <typename E1, typename E2>
struct scalar_type<MatrixAddition<E1, E2>>
    : public set_type<decltype(std::declval<typename scalar_type<E1>::type>() + std::declval<typename scalar_type<E2>::type>())> {};

template <typename E1, typename E2>
struct scalar_type<MatrixSubtraction<E1, E2>>
    : public set_type<decltype(std::declval<typename scalar_type<E1>::type>() - std::declval<typename scalar_type<E2>::type>())> {};

template <typename E>
struct scalar_type<MatrixNegation<E>> : public set_type<typename scalar_type<E>::type> {};

template <typename E>
struct scalar_type<MatrixTranspose<E>> : public set_type<typename scalar_type<E>::type> {};

template <typename E, typename U>
struct scalar_type<MatrixScalarMultiplication<E, U>> : public set_type<decltype(std::declval<typename scalar_type<E>::type>() * std::declval<U>())> {};

template <typename T>
struct scalar_type<Matrix<T>> : public set_type<T> {};

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
    : public and_value<vectorizable<E1>::value, vectorizable<E2>::value, std::is_arithmetic<scalar_type_t<MatrixAddition<E1, E2>>>::value> {};

template <typename E1, typename E2>
struct vectorizable<MatrixSubtraction<E1, E2>>
    : public and_value<vectorizable<E1>::value, vectorizable<E2>::value, std::is_arithmetic<scalar_type_t<MatrixSubtraction<E1, E2>>>::value> {};

template <typename E>
struct vectorizable<MatrixNegation<E>> : public vectorizable<E> {};

template <typename E>
struct vectorizable<MatrixTranspose<E>> : public std::false_type {};

template <typename E, typename U>
struct vectorizable<MatrixScalarMultiplication<E, U>> : public and_value<vectorizable<E>::value, std::is_same<scalar_type_t<E>, U>::value> {};

template <typename T>
struct vectorizable<Matrix<T>> : public std::is_arithmetic<T> {};

template <typename Expr>
constexpr bool vectorizable_v = vectorizable<Expr>::value;


template <typename E>
struct needs_reference : public set_type<E> {};

template <typename T>
struct needs_reference<Matrix<T>> : public set_type<Matrix<T>&> {};

template <typename E>
using needs_reference_t = typename needs_reference<E>::type;


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

template <class...>
using void_t = void;

template <typename T>
struct type_consistent : public and_value<std::is_same<T, decltype(std::declval<T>() * std::declval<T>())>::value> {};

template <typename T>
constexpr bool type_consistent_v = type_consistent<T>::value;

/*
 *
 * Computation Tree
 * Transforms operations to be calculated more efficient
 */

/*
 * A^T^T = A
 * (AB)C = A(BC) check number of multiplications
 * (AB)^T = B^T * A^T first one is probably faster
 * A(B + C) = AB + AC
 * (B + C)D = BD + CD first is probably faster, but this check might be hard (we have to know that 'D' is the same matrix on both sites
 */

// TODO: do this at compile time?
// TODO: apply simplify recursively

template <typename E>
MatrixExpression<E> simplify(MatrixExpression<E> e) {
	return e;
}

template <typename E1, typename E2>
MatrixAddition<E1, E2> simplify(MatrixAddition<E1, E2> e) {
	return e;
}

template <typename E1, typename E2>
MatrixSubtraction<E1, E2> simplify(MatrixSubtraction<E1, E2> e) {
	return e;
}

template <typename E>
MatrixNegation<E> simplify(MatrixNegation<E> e) {
	return e;
}

template <typename E>
MatrixTranspose<E> simplify(MatrixTranspose<E> e) {
	return e;
}

template <typename E, typename U>
MatrixScalarMultiplication<E, U> simplify(MatrixScalarMultiplication<E, U> e) {
	return e;
}

// What we really simplify
template <typename E>
E simplify(MatrixTranspose<MatrixTranspose<E>> e) {
	return e.getExpression().getExpression();
}

// template <typename E>
// MatrixScalarMultiplication<MatrixScalarMultiplication<E>> simplify(MatrixScalarMultiplication<MatrixScalarMultiplication<E>> e) {
//	// TODO: e.g. 5 * (6 * A) = 30 * A
//	return e;
//}

namespace detail {
template <int Depth = 1024, typename T>
void strassen_rec(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, coordinate_type size) {
	static_assert(Depth > 0, "strassen depth has to be > 0");
	if(size <= Depth) {
		matrix_multiplication(C, A, B);
		return;
	}

	coordinate_type m = size / 2;
	point_type size_m{m, m};

	const Matrix<T> a11 = A.sub({0, 0}, size_m);
	const Matrix<T> a12 = A.sub({0, m}, size_m);
	const Matrix<T> a21 = A.sub({m, 0}, size_m);
	const Matrix<T> a22 = A.sub(size_m, size_m);

	const Matrix<T> b11 = B.sub({0, 0}, size_m);
	const Matrix<T> b12 = B.sub({0, m}, size_m);
	const Matrix<T> b21 = B.sub({m, 0}, size_m);
	const Matrix<T> b22 = B.sub(size_m, size_m);

	Matrix<T> c11 = C.sub({0, 0}, size_m);
	Matrix<T> c12 = C.sub({0, m}, size_m);
	Matrix<T> c21 = C.sub({m, 0}, size_m);
	Matrix<T> c22 = C.sub(size_m, size_m);

	Matrix<T> u1(size_m);
	Matrix<T> u2(size_m);
	Matrix<T> u3(size_m);
	Matrix<T> u4(size_m);
	Matrix<T> u5(size_m);
	Matrix<T> u6(size_m);
	Matrix<T> u7(size_m);

	Matrix<T> s1(size_m);
	Matrix<T> s2(size_m);
	Matrix<T> s3(size_m);
	Matrix<T> s4(size_m);

	Matrix<T> t1(size_m);
	Matrix<T> t2(size_m);
	Matrix<T> t3(size_m);
	Matrix<T> t4(size_m);

	Matrix<T> p1(size_m);
	Matrix<T> p2(size_m);
	Matrix<T> p3(size_m);
	Matrix<T> p4(size_m);
	Matrix<T> p5(size_m);
	Matrix<T> p6(size_m);
	Matrix<T> p7(size_m);

	s1 = a21 + a22;
	s2 = s1 - a11;
	s3 = a11 - a21;
	s4 = a12 - s2;

	t1 = b12 - b11;
	t2 = b22 - t1;
	t3 = b22 - b12;
	t4 = t2 - b21;

	auto p1_async = algorithm::async([&]() { strassen_rec(a11, b11, p1, m); });

	auto p2_async = algorithm::async([&]() { strassen_rec(a12, b21, p2, m); });

	auto p3_async = algorithm::async([&]() { strassen_rec(s4, b22, p3, m); });

	auto p4_async = algorithm::async([&]() { strassen_rec(a22, t4, p4, m); });

	auto p5_async = algorithm::async([&]() { strassen_rec(s1, t1, p5, m); });

	auto p6_async = algorithm::async([&]() { strassen_rec(s2, t2, p6, m); });

	auto p7_async = algorithm::async([&]() { strassen_rec(s3, t3, p7, m); });

	p1_async.wait();
	p2_async.wait();
	p3_async.wait();
	p4_async.wait();
	p5_async.wait();
	p6_async.wait();
	p7_async.wait();

	u1 = p1 + p2;
	u2 = p1 + p6;
	u3 = u2 + p7;
	u4 = u2 + p5;
	u5 = u4 + p3;
	u6 = u3 - p4;
	u7 = u3 + p5;

	algorithm::pfor(size_m, [&](const point_type& p) {
		C[p] = u1[p];
		C[{p[0], p[1] + m}] = u5[p];
		C[{p[0] + m, p[1]}] = u6[p];
		C[{p[0] + m, p[1] + m}] = u7[p];
	});
}
}

struct RowRange {
	coordinate_type start;
	coordinate_type end;
};

template <typename E1, typename E2>
class MatrixAddition : public MatrixExpression<MatrixAddition<E1, E2>> {
	using typename MatrixExpression<MatrixAddition<E1, E2>>::T;
	using typename MatrixExpression<MatrixAddition<E1, E2>>::PacketScalar;

	using Exp1 = needs_reference_t<std::add_const_t<E1>>;
	using Exp2 = needs_reference_t<std::add_const_t<E2>>;

  public:
	MatrixAddition(Exp1 u, Exp2 v) : lhs(u), rhs(v) { assert(u.size() == v.size()); }

	T operator[](const point_type& pos) const { return lhs[pos] + rhs[pos]; }

	point_type size() const { return lhs.size(); }

	coordinate_type rows() const { return lhs.rows(); }

	coordinate_type columns() const { return lhs.columns(); }

	PacketScalar packet(point_type p) const { return lhs.packet(p) + rhs.packet(p); }

  private:
	Exp1 lhs;
	Exp2 rhs;
};

template <typename E1, typename E2>
class MatrixSubtraction : public MatrixExpression<MatrixSubtraction<E1, E2>> {
	using typename MatrixExpression<MatrixSubtraction<E1, E2>>::T;
	using typename MatrixExpression<MatrixSubtraction<E1, E2>>::PacketScalar;

	using Exp1 = needs_reference_t<std::add_const_t<E1>>;
	using Exp2 = needs_reference_t<std::add_const_t<E2>>;

  public:
	MatrixSubtraction(Exp1 u, const Exp2 v) : lhs(u), rhs(v) { assert_eq(lhs.size(), rhs.size()); }

	T operator[](const point_type& pos) const { return lhs[pos] - rhs[pos]; }

	point_type size() const { return rhs.size(); }

	coordinate_type rows() const { return lhs.rows(); }

	coordinate_type columns() const { return lhs.columns(); }

	PacketScalar packet(point_type p) const { return lhs.packet(p) - rhs.packet(p); }

  private:
	Exp1 lhs;
	Exp2 rhs;
};

template <typename E>
class MatrixTranspose : public MatrixExpression<MatrixTranspose<E>> {
	using typename MatrixExpression<MatrixTranspose<E>>::T;
	using typename MatrixExpression<MatrixTranspose<E>>::PacketScalar;

	using Exp = needs_reference_t<std::add_const_t<E>>;

  public:
	MatrixTranspose(Exp u) : expression(u) {}

	T operator[](const point_type& pos) const { return expression[{pos.y, pos.x}]; }

	point_type size() const { return {rows(), columns()}; }

	coordinate_type rows() const { return expression.columns(); }

	coordinate_type columns() const { return expression.rows(); }

	Exp getExpression() const { return expression; }

	//	PacketScalar packet(point_type p) const { assert_falsereturn expression.packet(p); }

  private:
	Exp expression;
};

template <typename E>
class MatrixNegation : public MatrixExpression<MatrixNegation<E>> {
	using typename MatrixExpression<MatrixNegation<E>>::T;
	using typename MatrixExpression<MatrixNegation<E>>::PacketScalar;

	using Exp = needs_reference_t<std::add_const_t<E>>;

  public:
	MatrixNegation(Exp e) : matrix(e) {}
	T operator[](const point_type& pos) const { return -matrix[pos]; }

	point_type size() const { return matrix.size(); }

	coordinate_type rows() const { return matrix.rows(); }

	coordinate_type columns() const { return matrix.columns(); }


	PacketScalar packet(point_type p) const { return -matrix.packet(p); }

  private:
	Exp matrix;
};


// TODO: also add a ScalarMatrixMultiplication class?
template <typename E, typename U>
class MatrixScalarMultiplication : public MatrixExpression<MatrixScalarMultiplication<E, U>> {
	using typename MatrixExpression<MatrixScalarMultiplication<E, U>>::T;
	using typename MatrixExpression<MatrixScalarMultiplication<E, U>>::PacketScalar;

	using Exp = needs_reference_t<std::add_const_t<E>>;

  public:
	MatrixScalarMultiplication(const U& u, Exp v) : scalar(u), matrix(v) {}

	T operator[](const point_type& pos) const { return scalar * matrix[pos]; }

	point_type size() const { return matrix.size(); }
	coordinate_type rows() const { return matrix.rows(); }

	coordinate_type columns() const { return matrix.columns(); }

	PacketScalar packet(point_type p) const { return matrix.packet(p) * PacketScalar(scalar); }

  private:
	const U& scalar;
	Exp matrix;
};

template <typename T>
class Matrix : public MatrixExpression<Matrix<T>> {
	using map_type = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
	using cmap_type = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
	using map_stride_type = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>>;
	using cmap_stride_type =
	    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>>;

	using typename MatrixExpression<Matrix<T>>::PacketScalar;

  public:
	Matrix(const point_type& size) : m_data(size) {}

	template <typename E>
	Matrix(const MatrixExpression<E>& mat) : m_data(mat.size()) {
		evaluate(mat, *this);
	}

	Matrix(const T& value) { fill(value); }

	template <typename Derived>
	Matrix(const Eigen::MatrixBase<Derived>& matrix) : m_data({matrix.rows(), matrix.cols()}) {
		algorithm::pfor(size(), [&](const point_type& p) { m_data[p] = matrix(p.x, p.y); });
	}

	Matrix(const Matrix& mat) : m_data(mat.size()) { evaluate(mat, *this); }

	Matrix(Matrix&&) = default;

	Matrix& operator=(const Matrix& mat) {
		evaluate(mat, *this);

		return *this;
	}

	Matrix& operator=(Matrix&&) = default;

	Matrix& operator=(const T& value) {
		fill(value);
		return (*this);
	}

	template <typename E>
	Matrix& operator=(MatrixExpression<E> const& mat) {
		evaluate(mat, *this);

		return *this;
	}

	inline T& operator[](const point_type& pos) { return m_data[pos]; }

	inline const T& operator[](const point_type& pos) const { return m_data[pos]; }

	inline point_type size() const { return m_data.size(); }

	inline coordinate_type rows() const { return m_data.size()[0]; }

	inline coordinate_type columns() const { return m_data.size()[1]; }

	map_stride_type sub(point_type start, point_type size) {
		return map_stride_type(&m_data[start], size.x, size.y, Eigen::OuterStride<Eigen::Dynamic>(columns()));
	}

	cmap_stride_type sub(point_type start, point_type size) const {
		return cmap_stride_type(&m_data[start], size.x, size.y, Eigen::OuterStride<Eigen::Dynamic>(columns()));
	}

	map_type sub(const RowRange& r) {
		assert_le(r.start, r.end);
		return map_type(&m_data[{r.start, 0}], r.end - r.start, columns());
	}

	cmap_type sub(const RowRange& r) const {
		assert_le(r.start, r.end);
		return cmap_type(&m_data[{r.start, 0}], r.end - r.start, columns());
	}

	void fill(const T& value) {
		algorithm::pfor(m_data.size(), [&](const point_type& p) { m_data[p] = value; });
	}

	void zero() { fill(0); }

	void identity() {
		assert_eq(rows(), columns());
		algorithm::pfor(m_data.size(), [&](const point_type& p) { m_data[p] = p[0] == p[1] ? 1. : 0.; });
	}

	template <typename Generator>
	void random(Generator gen) {
		// we do not use pfor here, rand() is not made for it
		for(coordinate_type i = 0; i < rows(); ++i) {
			for(coordinate_type j = 0; j < columns(); ++j) {
				m_data[{i, j}] = gen();
			}
		}
	}

	// Eigen matrices are stored column major by default
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> toEigenMatrix() {
		Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(rows(), columns());
		algorithm::pfor(size(), [&](const point_type& p) { result(p.x, p.y) = m_data[p]; });
		return result;
	}

	map_type getEigenMap() { return sub({0, rows()}); }

	cmap_type getEigenMap() const { return sub({0, rows()}); }

	PacketScalar packet(point_type p) const { return PacketScalar(&operator[](p)); }

  private:
	data::Grid<T, 2> m_data;
};

template <typename E>
class MatrixExpression {
  public:
	using T = scalar_type_t<E>;
	using PacketScalar = typename Vc::Vector<T>;

	T operator[](const point_type& pos) const { return static_cast<const E&>(*this)[pos]; }

	T at(const point_type& pos) const {
		assert_lt(pos, size());
		assert_ge(pos, (point_type{0, 0}));
		return static_cast<const E&>(*this)[pos];
	}

	point_type size() const { return static_cast<const E&>(*this).size(); }

	coordinate_type rows() const { return static_cast<const E&>(*this).rows(); }

	coordinate_type columns() const { return static_cast<const E&>(*this).columns(); }

	bool isSquare() const { return rows() == columns(); }

	MatrixTranspose<E> transpose() const { return MatrixTranspose<E>(static_cast<const E&>(*this)); }

	PacketScalar packet(point_type p) const { return static_cast<const E&>(*this).packet(p); }

	operator E&() { return static_cast<E&>(*this); }
	operator const E&() const { return static_cast<const E&>(*this); }
};

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


template <typename E1, typename E2>
bool isAlmostEqual(const MatrixExpression<E1>& a, const MatrixExpression<E2>& b, scalar_type_t<E1> epsilon = 0.001) {
	if(a.size()[0] != b.size()[0] || a.size()[1] != b.size()[1]) { return false; }
	for(coordinate_type i = 0; i < a.rows(); ++i) {
		for(coordinate_type j = 0; j < a.columns(); ++j) {
			scalar_type_t<E1> diff = (a[{i, j}] - b[{i, j}]);
			if(diff * diff > epsilon) { return false; }
		}
	}
	return true;
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

template <typename E>
Matrix<scalar_type_t<E>> eval(const MatrixExpression<E>& me) {
	Matrix<scalar_type_t<E>> tmp(me.size());

	evaluate(me, tmp);

	return tmp;
}

template <typename T>
const Matrix<T>& eval(const MatrixExpression<Matrix<T>>& me) {
	return me;
}

// -- evaluate a matrix expression using vectorization
template <typename E>
std::enable_if_t<vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, Matrix<scalar_type_t<E>>& dst) {
	assert_eq(expr.size(), dst.size());

	using T = scalar_type_t<E>;
	using PacketScalar = typename Vc::Vector<T>;

	const int total_size = expr.rows() * expr.columns();
	const int packet_size = PacketScalar::Size;
	const int aligned_end = total_size / packet_size * packet_size;

	algorithm::pfor(utils::Vector<coordinate_type, 1>(0), utils::Vector<coordinate_type, 1>(aligned_end / packet_size), [&](const auto& coord) {
		int i = coord[0] * packet_size;
		point_type p{i / expr.columns(), i - i / expr.columns() * expr.columns()};
		expr.packet(p).store(&dst[p]);
	});

	for(int i = aligned_end; i < total_size; i++) {
		point_type p{i / expr.columns(), i - i / expr.columns() * expr.columns()};
		dst[p] = expr[p];
	}
}

// -- evaluate a matrix expression by simply copying each value
template <typename E>
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expr, Matrix<scalar_type_t<E>>& dst) {
	assert_eq(expr.size(), dst.size());
	algorithm::pfor(expr.size(), [&](const auto& pos) { dst[pos] = expr[pos]; });
}

#define mindex(i, j, size) ((i) * (size) + (j))

// calculate a size * size block
template <int size = 8, typename T>
inline void block(point_type end, T* result, const T* lhs, const T* rhs, triple_type matrix_sizes) {
	using ct = coordinate_type;
	using vt = Vc::Vector<T>;

	static_assert(size % vt::Size == 0, "vector type size doesn't divide 'size'"); // our vector type 'vt' fits into the size x size segment

	constexpr int vector_size = size / vt::Size; // vector_size contains the number of vt types needed per line

	const auto m = matrix_sizes.x;
	const auto k = end.x;

	std::array<const T*, size> lhs_ptr;

	for(ct j = 0; j < size; ++j) {
		lhs_ptr[j] = lhs + mindex(j, 0, matrix_sizes.y);
	}

	std::array<std::array<vt, vector_size>, size> res;

	for(ct i = 0; i < k; ++i) {
		std::array<vt, size> a;

		for(ct j = 0; j < size; ++j) {
			a[j] = *lhs_ptr[j]++;
		}

		std::array<vt, vector_size> b;

		for(ct j = 0; j < vector_size; ++j) {
			b[j].load(rhs + j * vt::Size + i * size);


			for(ct jj = 0; jj < size; ++jj) {
				res[jj][j] += a[jj] * b[j];
			}
		}
	}

	for(ct i = 0; i < size; ++i) {
		for(ct j = 0; j < vector_size; ++j) {
			ct jj = j * (ct)vt::Size;
			for(ct k = 0; k < vt::Size; ++k) {
				result[mindex(i, jj + k, matrix_sizes.z)] += res[i][j][k];
			}
		}
	}
}

// -- parallel matrix * matrix multiplication kernel
template <int size = 8, typename T>
void kernel(point_type end, T* result, const T* lhs, const T* rhs, triple_type matrix_sizes) {
	using ct = coordinate_type;

	T packed_b[end.y * end.x];

	algorithm::pfor(GridPoint<1>{end.y / size}, [&](const auto& pos) {
		ct j = pos[0] * size;
		T* b_pos = packed_b + (j * end.x);
		for(int k = 0; k < end.x; ++k) {
			for(int jj = 0; jj < size; ++jj) {
				*b_pos++ = rhs[mindex(k, jj + j, matrix_sizes.z)];
			}
		}
	});

	algorithm::pfor(point_type{matrix_sizes.x / size, end.y / size}, [&](const auto& pos) {
		ct i = pos.x * size;
		ct j = pos.y * size;

		block<size>(end, result + mindex(i, j, matrix_sizes.z), lhs + mindex(i, 0, matrix_sizes.y), packed_b + (j * end.x), matrix_sizes);
	});

	for(ct i = 0; i < matrix_sizes.x - (matrix_sizes.x % size); ++i) {
		for(ct j = end.y - (end.y % size); j < end.y; ++j) {
			for(ct k = 0; k < end.x; ++k) {
				result[mindex(i, j, matrix_sizes.z)] += lhs[mindex(i, k, matrix_sizes.y)] * rhs[mindex(k, j, matrix_sizes.z)];
			}
		}
	}

	for(ct i = matrix_sizes.x - (matrix_sizes.x % size); i < matrix_sizes.x; ++i) {
		for(ct j = 0; j < end.y; ++j) {
			for(ct k = 0; k < end.x; ++k) {
				result[mindex(i, j, matrix_sizes.z)] += lhs[mindex(i, k, matrix_sizes.y)] * rhs[mindex(k, j, matrix_sizes.z)];
			}
		}
	}
}

// -- parallel matrix * matrix multiplication
template <typename T>
void matrix_multiplication_allscale(Matrix<T>& result, const Matrix<T>& lhs, const Matrix<T>& rhs) {
	assert(lhs.columns() == rhs.rows());

	using ct = coordinate_type;

	const coordinate_type nc = 512;
	const coordinate_type kc = 256;

	const auto m = lhs.rows();
	const auto k = lhs.columns();
	const auto n = rhs.columns();

	constexpr auto size = Vc::Vector<T>::Size;

	// TODO: find good values for kc, nc (multiple of vector size?)

	result.zero();


	for(ct kk = 0; kk < k; kk += kc) {
		ct kb = std::min(k - kk, kc);
		for(ct j = 0; j < n; j += nc) {
			ct jb = std::min(n - j, nc);

			kernel<size>({kb, jb}, &result[{0, j}], &lhs[{0, kk}], &rhs[{kk, j}], {m, k, n});
		}
	}
}

// -- matrix * matrix multiplication using a single BLAS level 3 function calls
void matrix_multiplication_blas(Matrix<double>& result, const Matrix<double>& lhs, const Matrix<double>& rhs) {
	assert(lhs.columns() == rhs.rows());

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lhs.rows(), rhs.columns(), lhs.columns(), 1.0, &lhs[{0, 0}], lhs.columns(), &rhs[{0, 0}],
	            rhs.columns(), 0.0, &result[{0, 0}], rhs.columns());
}

// -- parallel matrix * matrix multiplication using BLAS level 3 function calls
void matrix_multiplication_pblas(Matrix<double>& result, const Matrix<double>& lhs, const Matrix<double>& rhs) {
	assert(lhs.columns() == rhs.rows());

	auto blas_multiplication = [&](const RowRange& r) {
		assert_le(r.start, r.end);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, r.end - r.start, rhs.columns(), lhs.columns(), 1.0, &lhs[{r.start, 0}], lhs.columns(),
		            &rhs[{0, 0}], rhs.columns(), 0.0, &result[{r.start, 0}], rhs.columns());
	};

	auto multiplication_rec = prec(
	    // base case test
	    [&](const RowRange& r) { return r.start + 64 >= r.end; },
	    // base case
	    blas_multiplication, pick(
	                             // parallel recursive split
	                             [&](const RowRange& r, const auto& rec) {
		                             int mid = r.start + (r.end - r.start) / 2;
		                             return parallel(rec({r.start, mid}), rec({mid, r.end}));
		                         },
	                             // BLAS multiplication if no further parallelism can be exploited
	                             [&](const RowRange& r, const auto&) {
		                             blas_multiplication(r);
		                             return done();
		                         }));

	multiplication_rec({0, lhs.rows()}).wait();
}

// -- parallel matrix * matrix multiplication using BLAS level 2 function calls
void matrix_multiplication_pvblas(Matrix<double>& result, const Matrix<double>& lhs, const Matrix<double>& rhs) {
	assert(lhs.columns() == rhs.rows());

	algorithm::pfor(utils::Vector<coordinate_type, 1>{lhs.rows()}, [&](const auto& pos) {
		cblas_dgemv(CblasRowMajor, CblasNoTrans, lhs.rows(), lhs.columns(), 1., &lhs[{0, 0}], lhs.columns(), &rhs[{pos[0], 0}], 1, 0, &result[{pos[0], 0}], 1);
	});
}

// -- parallel matrix * matrix multiplication using the Eigen multiplication as base case
template <typename T>
void matrix_multiplication(Matrix<T>& result, const Matrix<T>& lhs, const Matrix<T>& rhs) {
	assert(lhs.columns() == rhs.rows());

	// create an Eigen map for the rhs of the multiplication
	auto eigen_rhs = rhs.getEigenMap();

	auto eigen_multiplication = [&](const RowRange& r) {
		assert_le(r.start, r.end);

		auto eigen_res_row = result.sub(r);
		auto eigen_lhs_row = lhs.sub(r);

		// Eigen matrix multiplication
		eigen_res_row = eigen_lhs_row * eigen_rhs;
	};

	auto multiplication_rec = prec(
	    // base case test
	    [&](const RowRange& r) { return r.start + 64 >= r.end; },
	    // base case
	    eigen_multiplication, pick(
	                              // parallel recursive split
	                              [&](const RowRange& r, const auto& rec) {
		                              int mid = r.start + (r.end - r.start) / 2;
		                              return parallel(rec({r.start, mid}), rec({mid, r.end}));
		                          },
	                              // Eigen multiplication if no further parallelism can be exploited
	                              [&](const RowRange& r, const auto&) {
		                              eigen_multiplication(r);
		                              return done();
		                          }));

	multiplication_rec({0, lhs.rows()}).wait();
}

// -- scalar * matrix multiplication
// Note: without the std::enable_if a matrix * matrix multiplication would be ambiguous
template <typename E, typename U>
std::enable_if_t<!std::is_base_of<MatrixExpression<U>, U>::value, MatrixScalarMultiplication<E, U>> operator*(const U& u, const MatrixExpression<E>& v) {
	return MatrixScalarMultiplication<E, U>(u, v);
}

template <typename E, typename U>
std::enable_if_t<!std::is_base_of<MatrixExpression<U>, U>::value, MatrixScalarMultiplication<E, U>> operator*(const MatrixExpression<E>& v, const U& u) {
	return u * v;
}

// -- matrix * matrix multiplication
template <typename E1, typename E2>
Matrix<scalar_type_t<E1>> operator*(const MatrixExpression<E1>& u, const MatrixExpression<E2>& v) {
	Matrix<scalar_type_t<E1>> tmp({u.rows(), v.columns()});
	matrix_multiplication(tmp, eval(u), eval(v));
	return tmp;
}


// -- Strassen-Winograd's matrix multiplication algorithm

template <int Depth = 2048, typename T>
Matrix<T> strassen(const Matrix<T>& A, const Matrix<T>& B) {
	assert_eq(A.columns(), B.rows());

	auto max = std::max({A.columns(), A.rows(), B.columns(), B.rows()});
	long m = std::pow(2, int(std::ceil(std::log2(max))));

	point_type size{m, m};

	if(A.size() == size && B.size() == size) {
		// no need to resize
		Matrix<T> result(size);

		detail::strassen_rec<Depth>(A, B, result, m);

		return result;
	} else {
		// resize and call the actual strassen algorithm
		Matrix<T> A_padded(size);
		Matrix<T> B_padded(size);

		algorithm::pfor(A.size(), [&](const point_type& p) {
			A_padded[p] = p[0] < A.rows() && p[1] < A.columns() ? A[p] : 0;
			B_padded[p] = p[0] < B.rows() && p[1] < B.columns() ? B[p] : 0;
		});

		Matrix<T> result_padded(size);

		detail::strassen_rec<Depth>(A_padded, B_padded, result_padded, m);

		Matrix<T> result({A.rows(), B.columns()});

		algorithm::pfor(result.size(), [&](const point_type& p) { result[p] = result_padded[p]; });

		return result;
	}
}

} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
