#pragma once

#include "traits.h"

#include "forward.h"

#include <Eigen/Dense>
#include <Vc/Vc>
#include <algorithm>
#include <allscale/api/user/data/grid.h>
#include <functional>

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

using point_type = GridPoint<2>;
using triple_type = GridPoint<3>;

struct RowRange {
	coordinate_type start;
	coordinate_type end;
};

struct BlockRange {
	point_type start;
	point_type end;

	point_type range() const { return end - start; }

	coordinate_type area() const {
		auto x = range();
		return x.x * x.y;
	}
};

template <typename E1, typename E2>
class MatrixAddition : public MatrixExpression<MatrixAddition<E1, E2>> {
	using typename MatrixExpression<MatrixAddition<E1, E2>>::T;
	using typename MatrixExpression<MatrixAddition<E1, E2>>::PacketScalar;

	using Exp1 = expression_member_t<E1>;
	using Exp2 = expression_member_t<E2>;

  public:
	MatrixAddition(Exp1 u, Exp2 v) : lhs(u), rhs(v) { assert(u.size() == v.size()); }

	T operator[](const point_type& pos) const { return lhs[pos] + rhs[pos]; }

	point_type size() const { return lhs.size(); }

	coordinate_type rows() const { return lhs.rows(); }

	coordinate_type columns() const { return lhs.columns(); }

	PacketScalar packet(point_type p) const { return lhs.packet(p) + rhs.packet(p); }

	Exp1 getLeftExpression() const { return lhs; }

	Exp2 getRightExpression() const { return rhs; }

  private:
	Exp1 lhs;
	Exp2 rhs;
};

template <typename E1, typename E2>
class MatrixSubtraction : public MatrixExpression<MatrixSubtraction<E1, E2>> {
	using typename MatrixExpression<MatrixSubtraction<E1, E2>>::T;
	using typename MatrixExpression<MatrixSubtraction<E1, E2>>::PacketScalar;

	using Exp1 = expression_member_t<E1>;
	using Exp2 = expression_member_t<E2>;

  public:
	MatrixSubtraction(Exp1 u, Exp2 v) : lhs(u), rhs(v) { assert_eq(lhs.size(), rhs.size()); }

	T operator[](const point_type& pos) const { return lhs[pos] - rhs[pos]; }

	point_type size() const { return rhs.size(); }

	coordinate_type rows() const { return lhs.rows(); }

	coordinate_type columns() const { return lhs.columns(); }

	PacketScalar packet(point_type p) const { return lhs.packet(p) - rhs.packet(p); }

	Exp1 getLeftExpression() const { return lhs; }

	Exp2 getRightExpression() const { return rhs; }

  private:
	Exp1 lhs;
	Exp2 rhs;
};

template <typename E1, typename E2>
class ElementMatrixMultiplication : public MatrixExpression<ElementMatrixMultiplication<E1, E2>> {
	using typename MatrixExpression<ElementMatrixMultiplication<E1, E2>>::T;
	using typename MatrixExpression<ElementMatrixMultiplication<E1, E2>>::PacketScalar;

	using Exp1 = expression_member_t<E1>;
	using Exp2 = expression_member_t<E2>;

  public:
	ElementMatrixMultiplication(Exp1 u, Exp2 v) : lhs(u), rhs(v) { assert_eq(lhs.size(), rhs.size()); }

	T operator[](const point_type& pos) const { return lhs[pos] * rhs[pos]; }

	point_type size() const { return lhs.size(); }

	coordinate_type rows() const { return lhs.rows(); }

	coordinate_type columns() const { return lhs.columns(); }

	PacketScalar packet(point_type p) const { return lhs.packet(p) * rhs.packet(p); }

	Exp1 getLeftExpression() const { return lhs; }

	Exp2 getRightExpression() const { return rhs; }

  private:
	Exp1 lhs;
	Exp2 rhs;
};

template <typename E1, typename E2>
class MatrixMultiplication : public MatrixExpression<MatrixMultiplication<E1, E2>> {
	using typename MatrixExpression<MatrixMultiplication<E1, E2>>::T;
	using typename MatrixExpression<MatrixMultiplication<E1, E2>>::PacketScalar;

	using Exp1 = expression_member_t<E1>;
	using Exp2 = expression_member_t<E2>;

  public:
	MatrixMultiplication(Exp1 u, Exp2 v) : lhs(u), rhs(v), tmp(nullptr) { assert_eq(lhs.columns(), rhs.rows()); }

	T operator[](const point_type& pos) const {
		evaluate();
		return (*tmp)[pos];
	}

	point_type size() const { return {rows(), columns()}; }

	coordinate_type rows() const { return lhs.rows(); }

	coordinate_type columns() const { return rhs.columns(); }

	PacketScalar packet(point_type p) const {
		evaluate();
		return tmp->packet(p);
	}

	Exp1 getLeftExpression() const { return lhs; }

	Exp2 getRightExpression() const { return rhs; }

	void evaluate() const {
		if(tmp != nullptr) return;

		tmp = std::make_shared<Matrix<T>>(size());

		matrix_multiplication(*tmp, lhs, rhs);
	}

  private:
	Exp1 lhs;
	Exp2 rhs;
	mutable std::shared_ptr<Matrix<T>> tmp; // contains a temporary matrix
};

template <typename E>
class MatrixTranspose : public MatrixExpression<MatrixTranspose<E>> {
	using typename MatrixExpression<MatrixTranspose<E>>::T;
	using typename MatrixExpression<MatrixTranspose<E>>::PacketScalar;

	using Exp = expression_member_t<E>;

  public:
	MatrixTranspose(Exp u) : expression(u) {}

	T operator[](const point_type& pos) const { return expression[{pos.y, pos.x}]; }

	point_type size() const { return {rows(), columns()}; }

	coordinate_type rows() const { return expression.columns(); }

	coordinate_type columns() const { return expression.rows(); }

	Exp getExpression() const { return expression; }

	//  PacketScalar packet(point_type p) const { assert_falsereturn expression.packet(p); }

  private:
	Exp expression;
};

template <typename E>
class MatrixNegation : public MatrixExpression<MatrixNegation<E>> {
	using typename MatrixExpression<MatrixNegation<E>>::T;
	using typename MatrixExpression<MatrixNegation<E>>::PacketScalar;

	using Exp = expression_member_t<E>;

  public:
	MatrixNegation(Exp e) : expression(e) {}
	T operator[](const point_type& pos) const { return -expression[pos]; }

	point_type size() const { return expression.size(); }

	coordinate_type rows() const { return expression.rows(); }

	coordinate_type columns() const { return expression.columns(); }

	PacketScalar packet(point_type p) const { return -expression.packet(p); }

	Exp getExpression() const { return expression; }

  private:
	Exp expression;
};

template <typename E, typename U>
class MatrixScalarMultiplication : public MatrixExpression<MatrixScalarMultiplication<E, U>> {
	using typename MatrixExpression<MatrixScalarMultiplication<E, U>>::T;
	using typename MatrixExpression<MatrixScalarMultiplication<E, U>>::PacketScalar;

	using Exp = expression_member_t<E>;

  public:
	MatrixScalarMultiplication(Exp v, const U& u) : scalar(u), expression(v) {}

	T operator[](const point_type& pos) const { return expression[pos] * scalar; }

	point_type size() const { return expression.size(); }
	coordinate_type rows() const { return expression.rows(); }

	coordinate_type columns() const { return expression.columns(); }

	PacketScalar packet(point_type p) const { return expression.packet(p) * PacketScalar(scalar); }

	const U& getScalar() const { return scalar; }

	Exp getExpression() const { return expression; }

  private:
	const U scalar;
	Exp expression;
};

template <typename E, typename U>
class ScalarMatrixMultiplication : public MatrixExpression<ScalarMatrixMultiplication<E, U>> {
	using typename MatrixExpression<ScalarMatrixMultiplication<E, U>>::T;
	using typename MatrixExpression<ScalarMatrixMultiplication<E, U>>::PacketScalar;

	using Exp = expression_member_t<E>;

  public:
	ScalarMatrixMultiplication(const U& u, Exp v) : scalar(u), expression(v) {}

	T operator[](const point_type& pos) const { return scalar * expression[pos]; }

	point_type size() const { return expression.size(); }
	coordinate_type rows() const { return expression.rows(); }

	coordinate_type columns() const { return expression.columns(); }

	PacketScalar packet(point_type p) const { return PacketScalar(scalar) * expression.packet(p); }

	const U& getScalar() const { return scalar; }

	Exp getExpression() const { return expression; }

  private:
	const U scalar;
	Exp expression;
};

template <typename T>
class Matrix : public MatrixExpression<Matrix<T>> {
	using map_type = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
	using cmap_type = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

	using typename MatrixExpression<Matrix<T>>::PacketScalar;

  public:
	Matrix(const point_type& size) : m_data(size) {}

	template <typename E>
	Matrix(const MatrixExpression<E>& mat) : m_data(mat.size()) {
		evaluate(mat, *this);
	}

	template <typename Derived>
	Matrix(const Eigen::MatrixBase<Derived>& matrix) : m_data({matrix.rows(), matrix.cols()}) {
		algorithm::pfor(size(), [&](const point_type& p) { m_data[p] = matrix(p.x, p.y); });
	}

	Matrix(const Matrix& mat) : MatrixExpression<Matrix<T>>(), m_data(mat.size()) { evaluate(mat, *this); }

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

	map_type eigenSub(const RowRange& r) {
		assert_le(r.start, r.end);
		return map_type(&m_data[{r.start, 0}], r.end - r.start, columns());
	}

	cmap_type eigenSub(const RowRange& r) const {
		assert_le(r.start, r.end);
		return cmap_type(&m_data[{r.start, 0}], r.end - r.start, columns());
	}

	void fill(const T& value) {
		algorithm::pfor(m_data.size(), [&](const point_type& p) { m_data[p] = value; });
	}

	void zero() { fill(static_cast<T>(0)); }

	void eye() {
		algorithm::pfor(m_data.size(), [&](const point_type& p) { m_data[p] = p[0] == p[1] ? static_cast<T>(1) : static_cast<T>(0); });
	}

	void identity() {
		assert_eq(rows(), columns());
		eye();
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

	map_type getEigenMap() { return eigenSub({0, rows()}); }

	cmap_type getEigenMap() const { return eigenSub({0, rows()}); }

	PacketScalar packet(point_type p) const { return PacketScalar(&operator[](p)); }

	const Matrix<T>& eval() const { return *this; }
	Matrix<T>& eval() { return *this; }

  private:
	data::Grid<T, 2> m_data;
};

template <typename E>
class SubMatrix : public MatrixExpression<SubMatrix<E>> {
	using typename MatrixExpression<SubMatrix<E>>::T;
	using typename MatrixExpression<SubMatrix<E>>::PacketScalar;

	using Exp = expression_member_t<E>;

  public:
	SubMatrix(Exp v, point_type sub_start, point_type sub_size) : expression(v), sub_start(sub_start), sub_size(sub_size) {
		assert_le(sub_start + sub_size, expression.size());
	}

	T operator[](const point_type& pos) const { return expression[pos + sub_start]; }

	point_type size() const { return sub_size; }
	coordinate_type rows() const { return sub_size[0]; }

	coordinate_type columns() const { return sub_size[1]; }

	//    PacketScalar packet(point_type p) const { }

	Exp getExpression() const { return expression; }

	point_type getStart() const { return sub_start; }

  private:
	Exp expression;
	point_type sub_start;
	point_type sub_size;
};

template <typename T>
class IdentityMatrix : public MatrixExpression<IdentityMatrix<T>> {
  public:
	IdentityMatrix(point_type matrix_size) : matrix_size(matrix_size) {}

	T operator[](const point_type& pos) const {
		assert_lt(pos, matrix_size);
		return pos.x == pos.y ? static_cast<T>(1) : static_cast<T>(0);
	}

	point_type size() const { return matrix_size; }

	coordinate_type rows() const { return matrix_size[0]; }

	coordinate_type columns() const { return matrix_size[1]; }

  private:
	point_type matrix_size;
};


template <typename T>
struct LUD {
	LUD(const Matrix<T>& A) : L(A.size()), U(A.size()) {
		using ct = coordinate_type;
		assert_eq(A.rows(), A.columns());

		ct n = A.rows();

		for(ct i = 0; i < n; ++i) {
			for(ct j = 0; j < n; ++j) {
				if(j < i) {
					L[{j, i}] = static_cast<T>(0);
				} else {
					L[{j, i}] = A[{j, i}];
					for(ct k = 0; k < i; ++k) {
						L[{j, i}] -= L[{j, k}] * U[{k, i}];
					}
				}
			}
			for(ct j = 0; j < n; ++j) {
				if(j < i) {
					U[{i, j}] = static_cast<T>(0);
				} else if(j == i) {
					U[{i, j}] = static_cast<T>(1);
				} else {
					U[{i, j}] = A[{i, j}] / L[{i, i}];
					for(ct k = 0; k < i; ++k) {
						U[{i, j}] -= L[{i, k}] * U[{k, j}] / L[{i, i}];
					}
				}
			}
		}
	}

	LUD(const LUD<T>&) = delete;
	LUD(LUD<T>&&) = default;

	LUD<T>& operator=(const LUD<T>&) = delete;
	LUD<T>& operator=(LUD<T>&&) = default;


	const Matrix<T>& lower() const { return L; }

	const Matrix<T>& upper() { return U; }

  private:
	Matrix<T> L;
	Matrix<T> U;
};

template <typename T>
struct QRD {
	QRD(const Matrix<T>& A) : Q(point_type{A.rows(), A.rows()}), R(A) {
		using ct = coordinate_type;

		// Householder QR Decomposition
		T mag, alpha;
		Matrix<T> u({A.rows(), 1});
		Matrix<T> v({A.rows(), 1});

		Matrix<T> P(point_type{A.rows(), A.rows()});
		Matrix<T> I(IdentityMatrix<T>(point_type{A.rows(), A.rows()})); // TODO: fix

		Q.identity();

		for(ct i = 0; i < A.columns(); ++i) {
			u.zero();
			v.zero();

			mag = 0;
			for(ct j = i; j < A.rows(); ++j) {
				u[{j, 0}] = R[{j, i}];
				mag += u[{j, 0}] * u[{j, 0}];
			}
			mag = std::sqrt(mag);

			alpha = u[{i, 0}] < 0 ? mag : -mag;

			mag = 0.0;
			for(ct j = i; j < A.rows(); ++j) {
				v[{j, 0}] = j == i ? u[{j, 0}] + alpha : u[{j, 0}];

				mag += v[{j, 0}] * v[{j, 0}];
			}
			mag = std::sqrt(mag);

			if(mag < 1E-10) continue;

			for(ct j = i; j < A.rows(); ++j) {
				v[{j, 0}] /= mag;
			}

			P = I - (v * v.transpose()) * 2.0;

			R = P * R;
			Q = Q * P;
		}
	}

	QRD(const QRD<T>&) = delete;
	QRD(QRD<T>&&) = default;

	QRD<T>& operator=(const QRD<T>&) = delete;
	QRD<T>& operator=(QRD<T>&&) = default;

	const Matrix<T>& getQ() const { return Q; }
	const Matrix<T>& getR() const { return R; }

  private:
	Matrix<T> Q;
	Matrix<T> R;
};

template <typename T>
struct SVD {
	SVD(const Matrix<T>& A) : U(point_type{A.rows(), A.rows()}), S(A.size()), V(point_type{A.columns(), A.columns()}) {
		// TODO: implement
		assert_fail();
	}

	SVD(const SVD<T>&) = delete;
	SVD(SVD<T>&&) = default;

	SVD<T>& operator=(const SVD<T>&) = delete;
	SVD<T>& operator=(SVD<T>&&) = default;

	const Matrix<T>& getU() { return U; }
	const Matrix<T>& getS() { return S; }
	const Matrix<T>& getV() { return V; }

  private:
	Matrix<T> U;
	Matrix<T> S;
	Matrix<T> V;
};


template <typename E>
class MatrixExpression {
  public:
	using T = scalar_type_t<E>;
	using PacketScalar = typename Vc::Vector<T>;

	/*
	 * abstract class due to object slicing
	 */
  protected:
	MatrixExpression() = default;
	MatrixExpression(const MatrixExpression&) = default;
	MatrixExpression& operator=(const MatrixExpression&) = delete;

  public:
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

	SubMatrix<E> row(coordinate_type r) const {
		assert_lt(r, rows());
		return sub({r, 0}, {1, columns()});
	}

	SubMatrix<E> column(coordinate_type c) const {
		assert_lt(c, columns());
		return sub({0, c}, {rows(), 1});
	}

	template <typename E2>
	ElementMatrixMultiplication<E, E2> product(const MatrixExpression<E2>& e) {
		return ElementMatrixMultiplication<E, E2>(static_cast<const E&>(*this), e);
	}

	MatrixTranspose<E> transpose() const { return MatrixTranspose<E>(static_cast<const E&>(*this)); }

	SubMatrix<E> sub(point_type start, point_type size) const { return SubMatrix<E>(static_cast<const E&>(*this), start, size); }

	T norm() { return std::sqrt(product(*this).reduce(0, std::plus<T>{})); }

	LUD<T> LUDecomposition() { return LUD<T>(*this); }

	QRD<T> QRDecomposition() { return QRD<T>(*this); }

	SVD<T> SVDecomposition() { return SVD<T>(*this); }

	template <typename Reducer>
	T reduce(T init, Reducer f) {
		using ct = coordinate_type;
		T result = init;
		// TODO: use preduce
		for(ct i = 0; i < rows(); ++i) {
			for(ct j = 0; j < columns(); ++j) {
				result = f(result, (*this)[{i, j}]);
			}
		}

		return result;
	}

	template <typename Reducer>
	T reduce(Reducer f) {
		using ct = coordinate_type;

		T result{};
		bool first = true;

		// TODO: use preduce
		for(ct i = 0; i < rows(); ++i) {
			for(ct j = 0; j < columns(); ++j) {
				if(first) {
					first = false;
					result = (*this)[{i, j}];
					continue;
				}
				result = f(result, (*this)[{i, j}]);
			}
		}

		return result;
	}

	T max() {
		return reduce([](T a, T b) { return std::max(a, b); });
	}

	T min() {
		return reduce([](T a, T b) { return std::min(a, b); });
	}

	T determinant() {
		assert_eq(rows(), columns());
		using ct = coordinate_type;
		auto lu = LUDecomposition();
		T det = static_cast<T>(1); // TODO: find a better way to do that

		const ct n = lu.lower().rows();

		for(ct i = 0; i < n; ++i) {
			det *= lu.lower()[{i, i}] * lu.upper()[{i, i}];
		}


		return det;
	}

	PacketScalar packet(point_type p) const { return static_cast<const E&>(*this).packet(p); }

	Matrix<T> eval() const {
		Matrix<T> tmp(size());

		evaluate(*this, tmp);

		return tmp;
	}

	operator E&() { return static_cast<E&>(*this); }
	operator const E&() const { return static_cast<const E&>(*this); }
};

// -- evaluate a matrix expression using vectorization
template <typename E>
std::enable_if_t<vectorizable_v<E>> evaluate(const MatrixExpression<E>& expression, Matrix<scalar_type_t<E>>& dst) {
	assert_eq(expression.size(), dst.size());

	expression_member_t<decltype(simplify(expression))> expr = simplify(expression);

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
template <typename E, typename T>
std::enable_if_t<!vectorizable_v<E>> evaluate(const MatrixExpression<E>& expression, Matrix<T>& dst) {
	assert_eq(expression.size(), dst.size());
	expression_member_t<decltype(simplify(expression))> expr = simplify(expression);

	algorithm::pfor(expr.size(), [&](const auto& pos) { dst[pos] = expr[pos]; });
}


template <typename T>
const Matrix<T>& simplify(const Matrix<T>& m) {
	return m;
}

template <typename T>
IdentityMatrix<T> simplify(IdentityMatrix<T> m) {
	return m;
}

template <typename E1, typename E2>
auto simplify(MatrixMultiplication<E1, E2> e) {
	MatrixMultiplication<std::decay_t<decltype(simplify(std::declval<E1>()))>, std::decay_t<decltype(simplify(std::declval<E2>()))>> e_simple(
	    simplify(e.getLeftExpression()), simplify(e.getRightExpression()));
	e_simple.evaluate();
	return e_simple;
}

template <typename T>
const Matrix<T>& simplify(const MatrixExpression<Matrix<T>>& e) {
	return static_cast<const Matrix<T>&>(e);
}

template <typename E>
auto simplify(const MatrixExpression<E>& e) {
	return simplify(static_cast<const E&>(e));
}

template <typename E1, typename E2>
auto simplify(MatrixAddition<E1, E2> e) {
	return MatrixAddition<std::decay_t<decltype(simplify(simplify(std::declval<E1>())))>, std::decay_t<decltype(simplify(std::declval<E2>()))>>(
	    simplify(e.getLeftExpression()), simplify(e.getRightExpression()));
}

template <typename E1, typename E2>
auto simplify(MatrixSubtraction<E1, E2> e) {
	return MatrixSubtraction<std::decay_t<decltype(simplify(simplify(std::declval<E1>())))>, std::decay_t<decltype(simplify(std::declval<E2>()))>>(
	    simplify(e.getLeftExpression()), simplify(e.getRightExpression()));
}

template <typename E1, typename E2>
auto simplify(ElementMatrixMultiplication<E1, E2> e) {
	return ElementMatrixMultiplication<std::decay_t<decltype(simplify(simplify(std::declval<E1>())))>, std::decay_t<decltype(simplify(std::declval<E2>()))>>(
	    simplify(e.getLeftExpression()), simplify(e.getRightExpression()));
}

template <typename E>
auto simplify(MatrixNegation<E> e) {
	return MatrixNegation<std::decay_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()));
}

template <typename E>
auto simplify(MatrixTranspose<E> e) {
	return MatrixTranspose<std::decay_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()));
}

template <typename E, typename U>
auto simplify(MatrixScalarMultiplication<E, U> e) {
	return MatrixScalarMultiplication<std::decay_t<decltype(simplify(std::declval<E>()))>, U>(simplify(e.getExpression()), e.getScalar());
}

template <typename E, typename U>
auto simplify(ScalarMatrixMultiplication<E, U> e) {
	return ScalarMatrixMultiplication<std::decay_t<decltype(simplify(std::declval<E>()))>, U>(e.getScalar(), simplify(e.getExpression()));
}

template <typename E>
auto simplify(SubMatrix<E> e) {
	return SubMatrix<std::decay_t<decltype(simplify(std::declval<E>()))>>(simplify(e.getExpression()), e.getStart(), e.size());
}

// What we really simplify
template <typename E>
expression_member_t<E> simplify(MatrixTranspose<MatrixTranspose<E>> e) {
	return e.getExpression().getExpression();
}

template <typename E>
expression_member_t<E> simplify(MatrixNegation<MatrixNegation<E>> e) {
	return e.getExpression().getExpression();
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_multiplication_v<U>, MatrixScalarMultiplication<E, U>>
simplify(MatrixScalarMultiplication<MatrixScalarMultiplication<E, U>, U> e) {
	return MatrixScalarMultiplication<E, U>(e.getExpression().getExpression(), e.getExpression().getScalar() * e.getScalar());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_multiplication_v<U>, ScalarMatrixMultiplication<E, U>>
simplify(ScalarMatrixMultiplication<MatrixScalarMultiplication<E, U>, U> e) {
	return ScalarMatrixMultiplication<E, U>(e.getExpression().getScalar() * e.getScalar(), e.getExpression().getExpression());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_multiplication_v<U>, ScalarMatrixMultiplication<E, U>>
simplify(ScalarMatrixMultiplication<ScalarMatrixMultiplication<E, U>, U> e) {
	return ScalarMatrixMultiplication<E, U>(e.getScalar() * e.getExpression().getScalar(), e.getExpression().getExpression());
}

template <typename E, typename U>
std::enable_if_t<is_associative_v<U> && std::is_same<U, scalar_type_t<E>>::value && type_consistent_multiplication_v<U>, MatrixScalarMultiplication<E, U>>
simplify(MatrixScalarMultiplication<ScalarMatrixMultiplication<E, U>, U> e) {
	return MatrixScalarMultiplication<E, U>(e.getExpression().getExpression(), e.getExpression().getScalar() * e.getScalar());
}

template <typename E, typename T>
expression_member_t<E> simplify(MatrixMultiplication<E, IdentityMatrix<T>> e) {
	assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
	return e.getLeftExpression();
}

template <typename E, typename T>
expression_member_t<E> simplify(MatrixMultiplication<IdentityMatrix<T>, E> e) {
	assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
	return e.getRightExpression();
}

template <typename T>
IdentityMatrix<T> simplify(MatrixMultiplication<IdentityMatrix<T>, IdentityMatrix<T>> e) {
	assert_eq(e.getLeftExpression().columns(), e.getRightExpression().rows());
	return e.getLeftExpression();
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
