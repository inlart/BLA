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
	 * Represents a part of a MatrixExpression E
	 */
	template<typename T>
    class SubMatrix;

	template<typename T>
	class Matrix;

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

    template<typename T>
    class SubMatrix : public MatrixExpression<SubMatrix<T>, T> {
    public:
        SubMatrix(const Matrix<T>& e, point_type start, point_type sub_size) : matrix(e), start(start), sub_size(sub_size) {
            assert_lt(start, e.size());
            assert_le(start + sub_size, e.size());
        }

        T& operator[](const point_type& pos) {
            assert_lt(pos, sub_size);
            return matrix[pos + start];
        }

        const T& operator[](const point_type& pos) const {
            assert_lt(pos, sub_size);
            return matrix[pos + start];
        }

        point_type size() const {
            return sub_size;
        }

        std::int64_t rows() const {
            return sub_size[0];
        }

        std::int64_t columns() const {
            return sub_size[1];
        }

        SubMatrix<T> sub(point_type start, point_type size) const {
            return SubMatrix<T>(matrix, this->start + start, size);
        }

    private:
        const Matrix<T>& matrix;
        const point_type start;
        const point_type sub_size;
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
            algorithm::pfor(point_type{m_data.size()},[&](const point_type& p) {
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

        SubMatrix<T> sub(point_type start, point_type size) const {
            return SubMatrix<T>(*this, start, size);
        }

        SubMatrix<T> sub() const {
            return sub({0, 0}, size());
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
            algorithm::pfor(point_type{m_data.size()},[&](const point_type& p) {
                    result(p.x, p.y) = m_data[p];
            });
            return result;
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


    /**
     * Due to the complexity of Matrix Multiplication, and the high
     * risk of aliasing problems, we handle it differently than other
     * matrix expressions: We force both parameters to be evaluated and
     * store the result in a temporary matrix.
     */
    template<typename T>
    void matmult_generic(T* result_data, const Matrix<T>& lhs, const Matrix<T>& rhs) {
        assert(lhs.columns() == rhs.rows()); //for multiplication to be valid this must hold
        /*static constexpr int strassen_cutoff = 2048; //if the result matrix would be larger in both dimensions than this, use strassen. TODO: test&tweak
        if(lhs.rows() > 2048 && rhs.columns() > 2048) {
            matmult_strassen(&m_tmp, u, v);
            return;
        }*/

        const T* lhs_data  = &lhs[{0,0}];
        const T* rhs_data  = &rhs[{0,0}];
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > eigen_rhs(rhs_data, rhs.rows(), rhs.columns()); //we take b fully

        struct Range {
            int start;
            int end;
        };

        auto handle_in_eigen = [&](const Range& r) {
            assert_le(r.start, r.end);
            T* result_row_ptr = result_data + lhs.columns() * r.start;
            const T* lhs_row_ptr = lhs_data + lhs.columns() * r.start;
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > eigen_res_row(result_row_ptr, r.end - r.start, rhs.columns()); //take r.end-r.start rows of "res"
            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > eigen_lhs_row(lhs_row_ptr, r.end - r.start, lhs.columns()); //take r.end-r.start rows of "a"
            eigen_res_row = eigen_lhs_row * eigen_rhs;
        };

        auto myLoop = prec(
            [&](const Range& r) { //base case test
                return r.start + 64 >= r.end; //we have a thin enough slice to want to hand over to eigen
            },
            handle_in_eigen,
            pick( //larger than base case handling
                [&](const Range& r, const auto& rec) { //here we split-up the work and call recursively
                    int mid = r.start + (r.end - r.start) / 2;
                    Range lhs = {r.start, mid};
                    Range rhs = {mid, r.end};
                    return parallel(rec(lhs), rec(rhs));
                },

                [&](const Range& r, const auto&) { //here we handle the whole block at once and do not recurse further
                    handle_in_eigen(r);
                    return done();
                }
            )
        );


        myLoop({0, (int)lhs.rows()}).wait();

    }

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

    template<typename E1, typename E2, typename T>
    Matrix<T> operator*(const MatrixExpression<E1, T>& u, const MatrixExpression<E2, T>& v) {
        Matrix<T> tmp({u.rows(), v.columns()});
        matmult_generic(&tmp[{0,0}], eval(u), eval(v));
        return tmp;
//        return strassen(eval(u), eval(v));
    }

    // strassen winograd
    template<typename T>
    void strassenR(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, coordinate_type size) {
        if (size <= 2048) {
            matmult_generic(&C[{0, 0}], A, B);
            //C[{0,0}] = A[{0,0}] * B[{0,0}];
            return;
        }

        coordinate_type m = size / 2;
        point_type size_m{m, m};

        /*
         * The output matrix C is expressed in terms of the block matrices M1..M7
         *
         * C1,1 = M1 + M4 - M5 + M7
         * C1,2 = M3 + M5
         * C2,1 = M2 + M4
         * C2,2 = M1 - M2 + M3 + M6
         *
         * Each of the block matrices M1..M7 is composed of quadrants from A and B as follows:
         *
         * M1 = (A1,1 + A2,2)(B1,1 + B2,2)
         * M2 = (A2,1 + A2,2)(B1,1)
         * M3 = (A1,1)(B1,2 - B2,2)
         * M4 = (A2,2)(B2,1 - B1,1)
         * M5 = (A1,1 + A1,2)(B2,2)
         * M6 = (A2,1 - A1,1)(B1,1 + B1,2)
         * M7 = (A1,2 - A2,2)(B2,1 + B2,2)
         */

//        SubMatrix<T> a11 = A.sub({0, 0}, size_m);
//        SubMatrix<T> a12 = A.sub({0, m}, size_m);
//        SubMatrix<T> a21 = A.sub({m, 0}, size_m);
//        SubMatrix<T> a22 = A.sub(size_m, size_m);
//
//        SubMatrix<T> b11 = B.sub({0, 0}, size_m);
//        SubMatrix<T> b12 = B.sub({0, m}, size_m);
//        SubMatrix<T> b21 = B.sub({m, 0}, size_m);
//        SubMatrix<T> b22 = B.sub(size_m, size_m);


        Matrix<T> a11 = A.sub({0, 0}, size_m);
        Matrix<T> a12 = A.sub({0, m}, size_m);
        Matrix<T> a21 = A.sub({m, 0}, size_m);
        Matrix<T> a22 = A.sub(size_m, size_m);

        Matrix<T> b11 = B.sub({0, 0}, size_m);
        Matrix<T> b12 = B.sub({0, m}, size_m);
        Matrix<T> b21 = B.sub({m, 0}, size_m);
        Matrix<T> b22 = B.sub(size_m, size_m);


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


        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a11_eigen(&a11[{0, 0}], a11.rows(), a11.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a12_eigen(&a12[{0, 0}], a12.rows(), a12.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a21_eigen(&a21[{0, 0}], a21.rows(), a21.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> a22_eigen(&a22[{0, 0}], a22.rows(), a22.columns());

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b11_eigen(&b11[{0, 0}], b11.rows(), b11.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b12_eigen(&b12[{0, 0}], b12.rows(), b12.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b21_eigen(&b21[{0, 0}], b21.rows(), b21.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> b22_eigen(&b22[{0, 0}], b22.rows(), b22.columns());


        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> s1_eigen(&s1[{0, 0}], s1.rows(), s1.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> s2_eigen(&s2[{0, 0}], s2.rows(), s2.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> s3_eigen(&s3[{0, 0}], s3.rows(), s3.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> s4_eigen(&s4[{0, 0}], s4.rows(), s4.columns());

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> t1_eigen(&t1[{0, 0}], t1.rows(), t1.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> t2_eigen(&t2[{0, 0}], t2.rows(), t2.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> t3_eigen(&t3[{0, 0}], t3.rows(), t3.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> t4_eigen(&t4[{0, 0}], t4.rows(), t4.columns());

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p1_eigen(&p1[{0, 0}], p1.rows(), p1.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p2_eigen(&p2[{0, 0}], p2.rows(), p2.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p3_eigen(&p3[{0, 0}], p3.rows(), p3.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p4_eigen(&p4[{0, 0}], p4.rows(), p4.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p5_eigen(&p5[{0, 0}], p5.rows(), p5.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p6_eigen(&p6[{0, 0}], p6.rows(), p6.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> p7_eigen(&p7[{0, 0}], p7.rows(), p7.columns());

        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u1_eigen(&u1[{0, 0}], u1.rows(), u1.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u2_eigen(&u2[{0, 0}], u2.rows(), u2.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u3_eigen(&u3[{0, 0}], u3.rows(), u3.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u4_eigen(&u4[{0, 0}], u4.rows(), u4.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u5_eigen(&u5[{0, 0}], u5.rows(), u5.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u6_eigen(&u6[{0, 0}], u6.rows(), u6.columns());
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u7_eigen(&u7[{0, 0}], u7.rows(), u7.columns());

        s1_eigen = a21_eigen + a22_eigen;
        s2_eigen = s1_eigen - a11_eigen;
        s3_eigen = a11_eigen - a21_eigen;
        s4_eigen = a12_eigen - s2_eigen;

        t1_eigen = b12_eigen - b11_eigen;
        t2_eigen = b22_eigen - t1_eigen;
        t3_eigen = b22_eigen - b12_eigen;
        t4_eigen = t2_eigen - b21_eigen;

        auto p1_async = algorithm::async([&](){
            strassenR(a11, b11, p1, m);
        });

        auto p2_async = algorithm::async([&](){
            strassenR(a12, b21, p2, m);
        });

        auto p3_async = algorithm::async([&](){
            strassenR(s4, b22, p3, m);
        });

        auto p4_async = algorithm::async([&](){
            strassenR(a22, t4, p4, m);
        });

        auto p5_async = algorithm::async([&](){
            strassenR(s1, t1, p5, m);
        });

        auto p6_async = algorithm::async([&](){
            strassenR(s2, t2, p6, m);
        });

        auto p7_async = algorithm::async([&](){
            strassenR(s3, t3, p7, m);
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

    template<typename T>
    Matrix<T> strassen(const Matrix<T>& A, const Matrix<T>& B) {
        assert_eq(A.columns(), B.rows());

        auto max = std::max({A.columns(), A.rows(), B.columns(), B.rows()});
        long m = std::pow(2, int(std::ceil(std::log2(max))));

        point_type size{m, m};

        //TODO: don't do this if A, B has the right size
        Matrix<T> A_padded(size);
        Matrix<T> B_padded(size);

        algorithm::pfor(A.size(), [&](const point_type& p) {
            A_padded[p] = p[0] < A.rows() && p[1] < A.columns() ? A[p] : 0;
            B_padded[p] = p[0] < B.rows() && p[1] < B.columns() ? B[p] : 0;
        });

        Matrix<T> result_padded(size);

        strassenR(A_padded, B_padded, result_padded, m);

        Matrix<T> result({A.rows(), B.columns()});

        algorithm::pfor(result.size(), [&](const point_type& p) {
            result[p] = result_padded[p];
        });

        return result;
    }

} //end namespace data
} //end namespace user
} //end namespace api
} //end namespace allscale
