#pragma once

#include <allscale/api/user/data/grid.h>
#include <allscale/api/user/algorithm/async.h>
#include <cstdlib>
#include <cmath>
#include <Eigen/Dense>


namespace allscale {
namespace api {
namespace user {
namespace data {
	using namespace allscale::api::core;

	using coordinate_type = std::int64_t;
	using point_type = GridPoint<2>;

	/*
	 * The base class for all matrix expressions
	 * Elements are not modifiable
	 */
	template<typename E, typename T>
	class MatrixExpression;

	/*
	 * Represents the sum of two MatrixExpressions E1 and E2
	 */
    template<typename E1, typename E2, typename T>
    class MatSum;

    /*
     * Represents the transposed MatrixExpression E
     */
	template<typename E, typename T>
	class MatrixTranspose;

	/*
	 * Represents the subtraction of MatrixExpressions E1 and E2
	 */
	template<typename E1, typename E2, typename T>
	class MatDiff;

	/*
	 * Represents the negation of the MatrixExpression E
	 */
	template<typename E, typename T>
	class NegMat;

    /*
     * Represents the Matrix
     * Elements are modifiable
     * Guarantees contiguous memory
     */
	template<typename T>
	class Matrix;

	namespace detail {
        template<int Depth = 2048, typename T>
        void strassen_rec(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, coordinate_type size) {
            if (size <= Depth) {
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

            auto a11_eigen = a11.getEigenMap();
            auto a12_eigen = a12.getEigenMap();
            auto a21_eigen = a21.getEigenMap();
            auto a22_eigen = a22.getEigenMap();

            auto b11_eigen = b11.getEigenMap();
            auto b12_eigen = b12.getEigenMap();
            auto b21_eigen = b21.getEigenMap();
            auto b22_eigen = b22.getEigenMap();

            auto s1_eigen = s1.getEigenMap();
            auto s2_eigen = s2.getEigenMap();
            auto s3_eigen = s3.getEigenMap();
            auto s4_eigen = s4.getEigenMap();

            auto t1_eigen = t1.getEigenMap();
            auto t2_eigen = t2.getEigenMap();
            auto t3_eigen = t3.getEigenMap();
            auto t4_eigen = t4.getEigenMap();

            auto p1_eigen = p1.getEigenMap();
            auto p2_eigen = p2.getEigenMap();
            auto p3_eigen = p3.getEigenMap();
            auto p4_eigen = p4.getEigenMap();
            auto p5_eigen = p5.getEigenMap();
            auto p6_eigen = p6.getEigenMap();
            auto p7_eigen = p7.getEigenMap();

            auto u1_eigen = u1.getEigenMap();
            auto u2_eigen = u2.getEigenMap();
            auto u3_eigen = u3.getEigenMap();
            auto u4_eigen = u4.getEigenMap();
            auto u5_eigen = u5.getEigenMap();
            auto u6_eigen = u6.getEigenMap();
            auto u7_eigen = u7.getEigenMap();

            s1_eigen = a21_eigen + a22_eigen;
            s2_eigen = s1_eigen - a11_eigen;
            s3_eigen = a11_eigen - a21_eigen;
            s4_eigen = a12_eigen - s2_eigen;

            t1_eigen = b12_eigen - b11_eigen;
            t2_eigen = b22_eigen - t1_eigen;
            t3_eigen = b22_eigen - b12_eigen;
            t4_eigen = t2_eigen - b21_eigen;

            auto p1_async = algorithm::async([&](){
                strassen_rec(a11, b11, p1, m);
            });

            auto p2_async = algorithm::async([&](){
                strassen_rec(a12, b21, p2, m);
            });

            auto p3_async = algorithm::async([&](){
                strassen_rec(s4, b22, p3, m);
            });

            auto p4_async = algorithm::async([&](){
                strassen_rec(a22, t4, p4, m);
            });

            auto p5_async = algorithm::async([&](){
                strassen_rec(s1, t1, p5, m);
            });

            auto p6_async = algorithm::async([&](){
                strassen_rec(s2, t2, p6, m);
            });

            auto p7_async = algorithm::async([&](){
                strassen_rec(s3, t3, p7, m);
            });

            p1_async.wait();
            p2_async.wait();
            p3_async.wait();
            p4_async.wait();
            p5_async.wait();
            p6_async.wait();
            p7_async.wait();

            u1_eigen = p1_eigen + p2_eigen;
            u2_eigen = p1_eigen + p6_eigen;
            u3_eigen = u2_eigen + p7_eigen;
            u4_eigen = u2_eigen + p5_eigen;
            u5_eigen = u4_eigen + p3_eigen;
            u6_eigen = u3_eigen - p4_eigen;
            u7_eigen = u3_eigen + p5_eigen;

            algorithm::pfor(size_m, [&](const point_type& p) {
                C[p] = u1[p];
                C[{p[0], p[1] + m}] = u5[p];
                C[{p[0] + m, p[1]}] = u6[p];
                C[{p[0] + m, p[1] + m}] = u7[p];
            });
        }
    }

    template<typename E, typename T>
    class MatrixExpression {
    public:

        T operator[](const point_type& pos) const {
            return static_cast<const E&>(*this)[pos];
        }

        point_type size() const {
        	return static_cast<const E&>(*this).size();
        }

        coordinate_type rows() const {
        	return static_cast<const E&>(*this).rows();
        }

        coordinate_type columns() const {
        	return static_cast<const E&>(*this).columns();
        }


		MatrixTranspose<E, T> transpose() const {
			return MatrixTranspose<E, T>(static_cast<const E&>(*this));
		}

        //these are used to unwrap the MatrixExpression into the actual subclass (e.g. MatSum)
        operator E& () { return static_cast<E&>(*this); }
        operator const E& () const { return static_cast<const E&>(*this); }
    };

    template<typename E1, typename E2, typename T>
    class MatSum : public MatrixExpression<MatSum<E1, E2, T>, T> {
    public:
        MatSum(const E1& u, const E2& v) : lhs(u), rhs(v) {
            assert(u.size() == v.size());
        }

        T operator[](const point_type& pos) const {
        	return lhs[pos] + rhs[pos];
        }

        point_type size() const {
        	return lhs.size();
        }

        std::int64_t rows() const {
        	return lhs.rows();
        }

        std::int64_t columns() const {
        	return lhs.columns();
        }

    private:
        const E1& lhs;
        const E2& rhs;
    };

    template<typename E1, typename E2, typename T>
    class MatDiff : public MatrixExpression<MatDiff<E1, E2, T>, T> {
    public:
        MatDiff(const E1& u, const E2& v) : lhs(u), rhs(v) {
            assert_eq(lhs.size(), rhs.size());
        }

        T operator[](const point_type& pos) const {
        	return lhs[pos] - rhs[pos];
        }

        point_type size() const {
        	return rhs.size();
        }

        std::int64_t rows() const {
        	return lhs.rows();
        }

        std::int64_t columns() const {
        	return lhs.columns();
        }

    private:
        const E1& lhs;
        const E2& rhs;
    };

    template<typename E, typename T>
    class MatrixTranspose : public MatrixExpression<MatrixTranspose<E, T>, T> {
    public:
		MatrixTranspose(const E& u) : expression(u) {}

		T operator[](const point_type& pos) const {
			return expression[{pos.y, pos.x}];
		}

		point_type size() const {
			return {rows(), columns()};
		}

		std::int64_t rows() const {
			return expression.columns();
		}

		std::int64_t columns() const {
			return expression.rows();
		}

	private:
		const E& expression;
	};

    template<typename E, typename T>
    class NegMat : public MatrixExpression<NegMat<E, T>, T> {
    public:
        NegMat(const E& e) : matrix(e) {}

        T operator[](const point_type& pos) const {
        	return -matrix[pos];
        }

        point_type size() const {
        	return matrix.size();
        }

        std::int64_t rows() const {
        	return matrix.rows();
        }
        std::int64_t columns() const {
        	return matrix.columns();
        }

    private:
        const E& matrix;
    };

    template <typename E1, typename E2, typename T>
    MatSum<E1,E2, T> const operator+(const MatrixExpression<E1, T>& u, const MatrixExpression<E2, T>& v) {
        return MatSum<E1, E2, T>(u, v);
    }

    template <typename E1, typename E2, typename T>
    MatDiff<E1, E2, T> const operator-(const MatrixExpression<E1, T>& u, const MatrixExpression<E2, T>& v) {
        return MatDiff<E1, E2, T>(u, v);
    }

    template <typename E, typename T>
    NegMat<E, T> const operator-(const MatrixExpression<E, T>& e) {
        return NegMat<E, T>(e);
    }

    template<typename T>
    class Matrix : public MatrixExpression<Matrix<T>, T> {
        using map_type = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>>;
        using cmap_type = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>>;
    public:
        Matrix(const point_type& size) : m_data(size) {}

        template<typename E>
        Matrix(MatrixExpression<E, T> const& mat) : m_data(mat.size()) {
            algorithm::pfor(point_type{m_data.size()},[&](const point_type& p) {
                    m_data[p] = mat[p];
            });
        }

        template<typename Derived>
        Matrix(const Eigen::MatrixBase<Derived>& x) : m_data({x.rows(), x.cols()}) {
            algorithm::pfor(size(), [&](const point_type& p) {
                m_data[p] = x(p.x, p.y);
            });
        }

        // enable move / disable copy
        Matrix(const Matrix&) = delete;
        Matrix(Matrix&&) = default;

        Matrix& operator=(const Matrix&) = delete;
        Matrix& operator=(Matrix&&) = default;

        T& operator[](const point_type& pos) {
        	return m_data[pos];
        }

        const T& operator[](const point_type& pos) const {
        	return m_data[pos];
        }

        point_type size() const {
        	return m_data.size();
        }

        std::int64_t rows() const {
        	return m_data.size()[0];
        }

        std::int64_t columns() const {
        	return m_data.size()[1];
        }

        map_type sub(point_type start, point_type size) {
            return map_type(&m_data[start], size.x, size.y, Eigen::OuterStride<Eigen::Dynamic>(columns()));
        }

        cmap_type sub(point_type start, point_type size) const {
            return cmap_type(&m_data[start], size.x, size.y, Eigen::OuterStride<Eigen::Dynamic>(columns()));
        }

        template<typename Op>
        void pforEach(const Op& op) {
            m_data.pforEach(op);
        }

        template<typename Op>
        void pforEach(const Op& op) const {
            m_data.pforEach(op);
        }

        template<typename Op>
        void forEach(const Op& op) {
            m_data.forEach(op);
        }

        template<typename Op>
        void forEach(const Op& op) const {
            m_data.forEach(op);
        }

        void zero() {
            algorithm::pfor(point_type{m_data.size()},[&](const point_type& p) {
                    m_data[p] = 0.0;
            });
        }

        void identity() {
            algorithm::pfor(point_type{m_data.size()},[&](const point_type& p) {
                m_data[p] = p[0] == p[1] ? 1. : 0.;
            });
        }

        template<typename Generator>
        void random(Generator gen) {
            //we do not use pfor here, rand() is not made for it
            for(std::int64_t i = 0; i < rows(); ++i) {
                for(std::int64_t j = 0; j < columns(); ++j) {
                    m_data[{i, j}] = gen();
                }
            }
        }

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> toEigenMatrix() {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result(rows(), columns());
            algorithm::pfor(size(),[&](const point_type& p) {
                result(p.x, p.y) = m_data[p];
            });
            return result;
        }

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> getEigenMap() {
            return Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(&m_data[{0, 0}], rows(), columns());
        }

        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> getEigenMap() const {
            return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(&m_data[{0, 0}], rows(), columns());
        }

    private:
        data::Grid<T, 2> m_data;
    };


    template<typename E1, typename E2, typename T>
    bool isAlmostEqual(const MatrixExpression<E1, T>& a, const MatrixExpression<E2, T>& b, T epsilon = 0.001) {
        if(a.size()[0] != b.size()[0] || a.size()[1] != b.size()[1]) {
            return false;
        }
        for(coordinate_type i = 0; i < a.rows(); ++i) {
            for(coordinate_type j = 0; j < a.columns(); ++j) {
                T diff = (a[{i, j}] - b[{i, j}]);
                if(diff*diff > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }

    template<typename E1, typename E2, typename T>
    bool operator==(const MatrixExpression<E1, T>& a, const MatrixExpression<E2, T>& b) {
        if(a.size() != b.size()) return false;

        for(coordinate_type i = 0; i < a.rows(); ++i) {
            for(coordinate_type j = 0; j < a.columns(); ++j) {
                if(a[{i, j}] != b[{i, j}]) return false;
            }
        }

        return true;
    }

    template<typename E1, typename E2, typename T>
    bool operator!=(const MatrixExpression<E1, T>& a, const MatrixExpression<E2, T>& b) {
        return !(a == b);
    }

    template<typename E, typename T>
    std::ostream &operator<<(std::ostream &os, MatrixExpression<E, T> const &m) {
        for(std::int64_t i = 0; i < m.rows(); ++i) {
            for(std::int64_t j = 0; j < m.columns(); ++j) {
                os << m[{i, j}];
                if(j != m.columns() - 1)
                    os << ", ";
            }
            if(i + 1 < m.rows()) {
                os << "\n";
            }
        }
        return os;
    }

    /**
     * The eval() function allows you to instruct an explicit evaluation of
     * any chained matrix operations, which is necessary to prevent aliasing
     * in some cases. It is used automatically in matrix multiplication.
     * If you eval() a Matrix<T>, a const reference to the matrix is returned,
     * if you eval any other MatrixExpression, a temporary Matrix will be created
     * and filled with the values of the evaluated MatrixExpression.
     */
    template<typename E, typename T>
    Matrix<T> eval(const MatrixExpression<E, T>& me) {
        assert(typeid(me) != typeid(MatrixExpression<Matrix<T>, T>)); //this is handled in the next function, a special case
        Matrix<T> tmp(me.size());
        algorithm::pfor(point_type{me.size()},[&](const point_type& p) {
                tmp[p] = me[p];
        });
        return tmp;
    }

    template<typename T>
    const Matrix<T>& eval(const MatrixExpression<Matrix<T>, T>& me) { //a performance optimisation, avoiding unneeded memory copying
        return me;
    }

    template<typename T>
    void matrix_multiplication(Matrix<T>& result, const Matrix<T>& lhs, const Matrix<T>& rhs) {
        assert(lhs.columns() == rhs.rows());

        struct Range {
            coordinate_type start;
            coordinate_type end;
        };

        // create an Eigen map for the rhs of the multiplication
        auto eigen_rhs = rhs.getEigenMap();

        auto eigen_multiplication = [&](const Range& r) {
            assert_le(r.start, r.end);

            auto eigen_res_row = result.sub({r.start, 0}, {r.end - r.start, result.columns()});
            auto eigen_lhs_row = lhs.sub({r.start, 0}, {r.end - r.start, lhs.columns()});

            // Eigen matrix multiplication
            eigen_res_row = eigen_lhs_row * eigen_rhs;
        };

        auto multiplication_rec = prec(
            // base case test
            [&](const Range& r) {
                return r.start + 64 >= r.end;
            },
            // base case
            eigen_multiplication,
            pick(
                // parallel recursive split
                [&](const Range& r, const auto& rec) {
                    int mid = r.start + (r.end - r.start) / 2;
                    return parallel(rec({r.start, mid}), rec({mid, r.end}));
                },
                // Eigen multiplication if no further parallelism can be exploited
                [&](const Range& r, const auto&) {
                    eigen_multiplication(r);
                    return done();
                }
            )
        );


        multiplication_rec({0, lhs.rows()}).wait();
    }

    /*
     * scalar * matrix multiplication
     */
    template<typename T>
    Matrix<T> operator*(const T u, const Matrix<T>& v) {
        Matrix<T> m(v.size());

        algorithm::pfor(m.size(), [&](point_type p) {
            m[p] = u * v[p];
        });

        return m;
    }

    template<typename T>
    Matrix<T> operator*(const Matrix<T>& v, const T u) {
        return u * v;
    }

    /*
     * matrix * matrix multiplication
     */
    template<typename E1, typename E2, typename T>
    Matrix<T> operator*(const MatrixExpression<E1, T>& u, const MatrixExpression<E2, T>& v) {
        Matrix<T> tmp({u.rows(), v.columns()});
        matrix_multiplication(tmp, eval(u), eval(v));
        return tmp;
    }

    /*
     * Strassen-Winograd's matrix multiplication algorithm
     */
    template<int Depth = 2048, typename T>
    Matrix<T> strassen(const Matrix<T>& A, const Matrix<T>& B) {
        assert_eq(A.columns(), B.rows());

        auto max = std::max({A.columns(), A.rows(), B.columns(), B.rows()});
        long m = std::pow(2, int(std::ceil(std::log2(max))));

        point_type size{m, m};

        if(A.size() == size && B.size() == size) {
            Matrix<T> result(size);

            detail::strassen_rec<Depth>(A, B, result, m);

            return result;
        }
        else {
            Matrix<T> A_padded(size);
            Matrix<T> B_padded(size);

            algorithm::pfor(A.size(), [&](const point_type& p) {
                A_padded[p] = p[0] < A.rows() && p[1] < A.columns() ? A[p] : 0;
                B_padded[p] = p[0] < B.rows() && p[1] < B.columns() ? B[p] : 0;
            });

            Matrix<T> result_padded(size);

            detail::strassen_rec<Depth>(A_padded, B_padded, result_padded, m);

            Matrix<T> result({A.rows(), B.columns()});

            algorithm::pfor(result.size(), [&](const point_type& p) {
                result[p] = result_padded[p];
            });

            return result;
        }
    }

} //end namespace data
} //end namespace user
} //end namespace api
} //end namespace allscale
