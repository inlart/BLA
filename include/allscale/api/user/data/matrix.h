#pragma once

#include <allscale/api/user/data/grid.h>
#include <cstdlib>
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


		MatrixTranspose<E, T> transpose() {
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

        void zero() {
            algorithm::pfor(point_type{m_data.size()},[&](const point_type& p) {
                    m_data[p] = 0.0;
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
        for(std::int64_t i = 0; i < a.rows(); ++i) {
            for(std::int64_t j = 0; j < a.columns(); ++j) {
                T diff = (a[{i, j}] - b[{i, j}]);
                if(diff*diff > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }

    template<typename E, typename T>
    std::ostream &operator<<(std::ostream &os, MatrixExpression<E, T> const &m) {
        for(std::int64_t i = 0; i < m.rows(); ++i) {
            for(std::int64_t j = 0; j < m.columns(); ++j) {
                os << m[{i, j}] << ", ";
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

    template<typename E1, typename E2, typename T>
    Matrix<T> operator*(const MatrixExpression<E1, T>& u, const MatrixExpression<E2, T>& v) {
        Matrix<T> tmp({u.rows(), v.columns()});
        matmult_generic(&tmp[{0,0}], eval(u), eval(v));
        return tmp;
    }
} //end namespace data
} //end namespace user
} //end namespace api
} //end namespace allscale
