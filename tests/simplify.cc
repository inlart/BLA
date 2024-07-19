#include <Vc/Vc>
#include <bla/matrix.h>
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <type_traits>

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
TEST(Simplify, Transpose) {
    Matrix<int> m1({55, 58});
    Matrix<int> m2({55, 58});
    Matrix<int> m3({55, 58});
    Matrix<int> m4({55, 58});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    m1.fill_seq(g);
    m2.fill_seq(g);
    m3.zero();
    m4.zero();

    m3 = m1 + m2;

    m4 = m1 + m2;

    ASSERT_EQ(m3, m4);

    m3 = (m1.transpose()).eval().transpose();
    m4 = simplify(m1.transpose().transpose());

    ASSERT_EQ(m3, m4);

    m3 = (m1 + m2).transpose().transpose();
    m4 = m1 + m2;

    ASSERT_EQ(m3, m4);
}

TEST(Simplify, RecursiveTranspose) {
    Matrix<int> m1({55, 58});
    Matrix<int> m2({55, 58});
    Matrix<int> m3({55, 58});
    Matrix<int> m4({55, 58});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    m1.fill_seq(g);
    m2.fill_seq(g);
    m3.zero();
    m4.zero();

    m3 = m1 + m2.transpose().transpose();

    m4 = simplify(m1 + m2.transpose().transpose());

    ASSERT_EQ(m3, m4);

    ASSERT_TRUE((std::is_same<std::decay_t<decltype(m1 + m2)>, std::decay_t<decltype(simplify(m1 + m2.transpose().transpose()))>>::value));
}

TEST(Simplify, MatrixScalarScalarMultiplication) {
    Matrix<int> m1({55, 55});
    Matrix<int> m2({55, 55});
    Matrix<int> m3({55, 55});

    m1.identity();
    m2.zero();
    m3.zero();

    m2 = m1 * 5 * 6;

    m3 = simplify(m1 * 5 * 6);

    ASSERT_EQ(m2, m3);

    ASSERT_TRUE((std::is_same<std::decay_t<decltype(m1 * 30)>, std::decay_t<decltype(simplify(m1 * 5 * 6))>>::value));
}

TEST(Simplify, ScalarScalarMatrixMultiplication) {
    Matrix<int> m1({55, 55});
    Matrix<int> m2({55, 55});
    Matrix<int> m3({55, 55});

    m1.identity();
    m2.zero();
    m3.zero();

    m2 = 5 * (6 * m1);


    m3 = simplify(5 * (6 * m1));

    ASSERT_EQ(m2, m3);

    ASSERT_TRUE((std::is_same<std::decay_t<decltype(30 * m1)>, std::decay_t<decltype(simplify(5 * (6 * m1)))>>::value));
}

TEST(Simplify, ScalarMatrixScalarMultiplication1) {
    Matrix<int> m1({55, 55});
    Matrix<int> m2({55, 55});
    Matrix<int> m3({55, 55});

    m1.identity();
    m2.zero();
    m3.zero();

    m2 = 6 * (m1 * 5);

    m3 = simplify(6 * (m1 * 5));

    ASSERT_EQ(m2, m3);

    ASSERT_TRUE((std::is_same<std::decay_t<decltype(m1)>, std::decay_t<decltype(simplify(6 * (m1 * 5)).getExpression())>>::value));
}

TEST(Simplify, ScalarMatrixScalarMultiplication2) {
    Matrix<int> m1({55, 55});
    Matrix<int> m2({55, 55});
    Matrix<int> m3({55, 55});

    m1.identity();
    m2.zero();
    m3.zero();

    m2 = (m1 * 5) * 6;

    m3 = simplify((m1 * 5) * 6);

    ASSERT_EQ(m2, m3);

    ASSERT_TRUE((std::is_same<std::decay_t<decltype(m1)>, std::decay_t<decltype(simplify((m1 * 5) * 6).getExpression())>>::value));
}

TEST(Simplify, Negation) {
    Matrix<int> m1({55, 58});
    Matrix<int> m2({55, 58});
    Matrix<int> m3({55, 58});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 9);

    auto g = [&](const auto&) { return dis(gen); };

    m1.fill_seq(g);
    m2.zero();
    m3.zero();

    m2 = -(-m1);

    m3 = simplify(-(-m1));


    ASSERT_EQ(m2, m3);


    ASSERT_TRUE((std::is_same<std::decay_t<decltype(m1)>, std::decay_t<decltype(simplify(-(-m1)))>>::value));
}

TEST(Simplify, IdentityMatrix) {
    Matrix<int> m1({55, 58});
    Matrix<int> m2({58, 55});
    IdentityMatrix<int> m3(point_type{55, 55});
    IdentityMatrix<int> m4(point_type{55, 55});

    Matrix<int> r1({55, 58});
    Matrix<int> r2({58, 55});
    Matrix<int> r3({55, 55});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 9);

    auto g = [&](const auto&) { return dis(gen); };

    m1.fill_seq(g);
    m2.fill_seq(g);

    r1 = simplify(m3 * m1);

    ASSERT_EQ(m1, r1);

    ASSERT_TRUE((std::is_same<std::decay_t<decltype(m1)>, std::decay_t<decltype(simplify(m3 * m1))>>::value));

    r2 = simplify(m2 * m3);

    ASSERT_EQ(m2, r2);

    ASSERT_TRUE((std::is_same<std::decay_t<decltype(m2)>, std::decay_t<decltype(simplify(m3 * m2))>>::value));

    r3 = simplify(m3 * m4);

    ASSERT_EQ(m3, r3);

    ASSERT_TRUE((std::is_same<std::decay_t<decltype(m3)>, std::decay_t<decltype(simplify(m3 * m4))>>::value));
}

TEST(Simplify, SubMatrixMultiplication) {
    Matrix<double> m1({55, 58});
    Matrix<double> m2({58, 55});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };

    m1.fill_seq(g);
    m2.fill_seq(g);

    BlockRange range({4, 7}, {13, 27});

    Matrix<double> result = m1 * m2;

    ASSERT_TRUE(isAlmostEqual(result.sub(range), (m1 * m2).sub(range)));
}

TEST(Simplify, SubMatrixAddition) {
    Matrix<double> m1({55, 58});
    Matrix<double> m2({55, 58});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };

    m1.fill_seq(g);
    m2.fill_seq(g);

    BlockRange range({4, 7}, {13, 27});

    Matrix<double> result = m1 + m2;

    ASSERT_TRUE(isAlmostEqual(result.sub(range), (m1 + m2).sub(range)));
}

TEST(Simplify, SubMatrixSubtraction) {
    Matrix<double> m1({55, 58});
    Matrix<double> m2({55, 58});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };

    m1.fill_seq(g);
    m2.fill_seq(g);

    BlockRange range({4, 7}, {13, 27});

    Matrix<double> result = m1 - m2;

    ASSERT_TRUE(isAlmostEqual(result.sub(range), (m1 - m2).sub(range)));
}

} // end namespace impl
} // namespace bla
