#include <bla/matrix.h>
#include <gtest/gtest.h>


namespace bla {
namespace impl {

template <typename E1, typename E2, typename T = double>
std::enable_if_t<!std::is_same<scalar_type_t<E1>, std::complex<double>>::value, bool> isAlmostEqual(const MatrixExpression<E1>& a,
                                                                                                    const MatrixExpression<E2>& b, T epsilon = 0.001) {
    if(a.size() != b.size()) {
        return false;
    }
    for(coordinate_type i = 0; i < a.rows(); ++i) {
        for(coordinate_type j = 0; j < a.columns(); ++j) {
            scalar_type_t<E1> diff = (a[{i, j}] - b[{i, j}]);
            if(diff * diff < epsilon) {
                continue;
            }
            return false;
        }
    }
    return true;
}

template <typename E1, typename E2, typename T = double>
std::enable_if_t<std::is_same<scalar_type_t<E1>, std::complex<double>>::value, bool> isAlmostEqual(const MatrixExpression<E1>& a, const MatrixExpression<E2>& b,
                                                                                                   T epsilon = 0.001) {
    if(a.size() != b.size()) {
        return false;
    }
    for(coordinate_type i = 0; i < a.rows(); ++i) {
        for(coordinate_type j = 0; j < a.columns(); ++j) {
            scalar_type_t<E1> diff = (a[{i, j}] - b[{i, j}]);
            if(diff.real() * diff.real() < epsilon && diff.imag() * diff.imag() < epsilon) {
                continue;
            }
            return false;
        }
    }
    return true;
}


TEST(Solve, EigenSolver) {
    Matrix<double> m1({4, 4});

    Matrix<double> zero({m1.rows(), 1});
    zero.zero();

    m1 << 52, 30, 49, 28, 30, 50, 8, 44, 49, 8, 46, 16, 28, 44, 16, 22;

    auto s = m1.solveEigen();

    for(unsigned int i = 0; i < s.eigenvalues.size(); ++i) {
        ASSERT_TRUE(isAlmostEqual(m1 * s.getEigenvector(i), s.getEigenvalue(i) * s.getEigenvector(i), 0.01));

        // we are searching for non-trivial solutions
        ASSERT_FALSE(isAlmostEqual(s.getEigenvector(i), zero));
    }
}

} // namespace impl
} // namespace bla
