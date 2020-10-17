#include <Vc/Vc>
#include <bla/matrix.h>
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <type_traits>

#include "utils.h"

namespace allscale {
namespace api {
namespace user {
namespace data {
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

TEST(Operation, Access) {
    Matrix<double> m({2, 2});
    m.zero();
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            ASSERT_EQ(0.0, (m[{i, j}]));
        }
    }
}

TEST(Operation, Equal) {
    Matrix<double> m1({45, 59});
    Matrix<double> m2({45, 59});

    for(int i = 0; i < 4; ++i) {
        m1.fill(1);
        m2.fill(1);
        ASSERT_EQ(m1, m2);
        ASSERT_EQ(m2, m1);

        m2 = 3. * m2;

        ASSERT_NE(m1, m2);
        ASSERT_NE(m2, m1);
    }
}

TEST(Operation, Addition) {
    Matrix<int> m1({123, 76});
    Matrix<int> m2(m1.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_EQ(m1 + m2, toAllscaleMatrix((toEigenMatrix(m1) + toEigenMatrix(m2)).eval()));
    }
}

TEST(Operation, AssignAddition) {
    Matrix<double> m1({123, 76});
    Matrix<double> m2(m1.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        Eigen::MatrixXd m1e = toEigenMatrix(m1);
        Eigen::MatrixXd m2e = toEigenMatrix(m2);

        m1 += m2;
        m1e += m2e;

        ASSERT_TRUE(isAlmostEqual(m1, toAllscaleMatrix(m1e)));
    }
}

TEST(Operation, AssignAdditionRefSubMatrix) {
    Matrix<double> m1({123, 76});
    Matrix<double> m2({5, 76});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        Eigen::MatrixXd m1e = toEigenMatrix(m1);
        Eigen::MatrixXd m2e = toEigenMatrix(m2);

        m1.sub({{0, 0}, {5, 76}}) += m2;
        m1e.block(0, 0, 5, 76) += m2e;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), Matrix<double>(toAllscaleMatrix((m1e.block(0, 0, 5, 76)).eval()))));
        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));

        m1.topRows(5) += m2;
        m1e.block(0, 0, 5, 76) += m2e;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), Matrix<double>(toAllscaleMatrix((m1e.block(0, 0, 5, 76)).eval()))));
        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));
    }
}

TEST(Operation, Subtraction) {
    Matrix<double> m1({31, 47});
    Matrix<double> m2(m1.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual(m1 - m2, Matrix<double>(toAllscaleMatrix((toEigenMatrix(m1) - toEigenMatrix(m2)).eval()))));
        ASSERT_TRUE(isAlmostEqual(m1 - m1, m2 - m2));
    }
}

TEST(Operation, AssignSubtraction) {
    Matrix<double> m1({123, 76});
    Matrix<double> m2(m1.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        Eigen::MatrixXd m1e = toEigenMatrix(m1);
        Eigen::MatrixXd m2e = toEigenMatrix(m2);

        m1 -= m2;
        m1e -= m2e;

        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));
    }
}

TEST(Operation, AssignSubtractionRefSubMatrix) {
    Matrix<double> m1({123, 76});
    Matrix<double> m2({5, 76});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        Eigen::MatrixXd m1e = toEigenMatrix(m1);
        Eigen::MatrixXd m2e = toEigenMatrix(m2);

        m1.sub({{0, 0}, {5, 76}}) -= m2;
        m1e.block(0, 0, 5, 76) -= m2e;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), Matrix<double>(toAllscaleMatrix(m1e.block(0, 0, 5, 76)))));
        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));

        m1.topRows(5) -= m2;
        m1e.block(0, 0, 5, 76) -= m2e;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), Matrix<double>(toAllscaleMatrix(m1e.block(0, 0, 5, 76)))));
        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));
    }
}

TEST(Operation, AssignScalarMultiplication) {
    Matrix<double> m1({123, 76});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        Eigen::MatrixXd m1e = toEigenMatrix(m1);

        auto number = g();

        m1 *= number;
        m1e *= number;

        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));
    }
}

TEST(Operation, AssignScalarMultiplicationRefSubMatrix) {
    Matrix<double> m1({123, 76});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        auto number = g();

        Eigen::MatrixXd m1e = toEigenMatrix(m1);

        m1.sub({{0, 0}, {5, 76}}) *= number;
        m1e.block(0, 0, 5, 76) *= number;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), Matrix<double>(toAllscaleMatrix(m1e.block(0, 0, 5, 76)))));
        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));

        number = g();

        m1.topRows(5) *= number;
        m1e.block(0, 0, 5, 76) *= number;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), Matrix<double>(toAllscaleMatrix(m1e.block(0, 0, 5, 76)))));
        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));
    }
}

TEST(Operation, AssignScalarDivision) {
    Matrix<double> m1({123, 76});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        Eigen::MatrixXd m1e = toEigenMatrix(m1);

        auto number = g();

        m1 /= number;
        m1e /= number;

        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));
    }
}

TEST(Operation, AssignScalarDivisionRefSubMatrix) {
    Matrix<double> m1({123, 76});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        auto number = g();

        Eigen::MatrixXd m1e = toEigenMatrix(m1);

        m1.sub({{0, 0}, {5, 76}}) /= number;
        m1e.block(0, 0, 5, 76) /= number;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), Matrix<double>(toAllscaleMatrix(m1e.block(0, 0, 5, 76)))));
        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));

        number = g();

        m1.topRows(5) /= number;
        m1e.block(0, 0, 5, 76) /= number;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), Matrix<double>(toAllscaleMatrix(m1e.block(0, 0, 5, 76)))));
        ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(toAllscaleMatrix(m1e))));
    }
}

TEST(Operation, Negation) {
    Matrix<double> m({100, 99});
    m.zero();
    ASSERT_TRUE(isAlmostEqual(m, -m));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual(-m, Matrix<double>(toAllscaleMatrix(-(toEigenMatrix(m))))));
    }
}


TEST(Operation, ScalarMatrixMultiplication) {
    Matrix<double> m1({45, 45});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        ASSERT_EQ(3. * m1, toAllscaleMatrix((3. * toEigenMatrix(m1)).eval()));
    }
}

TEST(Operation, MatrixScalarMultiplication) {
    Matrix<double> m1({45, 45});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        ASSERT_EQ(m1 * 3., toAllscaleMatrix((toEigenMatrix(m1) * 3.).eval()));
    }
}

TEST(Operation, Multiple) {
    Matrix<double> m1({55, 55});
    Matrix<double> m2({55, 56});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m1e = toEigenMatrix(m1);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m2e = toEigenMatrix(m2);
        ASSERT_TRUE(isAlmostEqual(-(m1 + m1) * m2 + m2 - m2 + m2 - m2, toAllscaleMatrix(-(m1e + m1e) * m2e + m2e - m2e + m2e - m2e)));
    }
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
