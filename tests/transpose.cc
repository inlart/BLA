#include <Vc/Vc>
#include <bla/matrix.h>
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <type_traits>

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

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

TEST(Operation, Transpose) {
    Matrix<double> m1({47, 39});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    m1.fill_seq(g);
    Matrix<double> m2 = m1.transpose();

    ASSERT_EQ(m1.rows(), m2.columns());
    ASSERT_EQ(m2.rows(), m1.columns());

    allscale::api::user::algorithm::pfor(m1.size(), [&](const point_type& p) { ASSERT_EQ(m1[p], (m2[{p.y, p.x}])); });
}

TEST(Operation, TransposeFloat) {
    Matrix<float> m1({47, 39});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    m1.fill_seq(g);
    Matrix<float> m2 = m1.transpose();

    ASSERT_EQ(m1.rows(), m2.columns());
    ASSERT_EQ(m2.rows(), m1.columns());

    allscale::api::user::algorithm::pfor(m1.size(), [&](const point_type& p) { ASSERT_EQ(m1[p], (m2[{p.y, p.x}])); });
}

TEST(Operation, TransposeInt) {
    Matrix<int> m1({47, 39});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 9);

    auto g = [&](const auto&) { return dis(gen); };
    m1.fill_seq(g);
    Matrix<int> m2 = m1.transpose();

    ASSERT_EQ(m1.rows(), m2.columns());
    ASSERT_EQ(m2.rows(), m1.columns());

    allscale::api::user::algorithm::pfor(m1.size(), [&](const point_type& p) { ASSERT_EQ(m1[p], (m2[{p.y, p.x}])); });
}

} // end namespace impl
} // namespace bla
