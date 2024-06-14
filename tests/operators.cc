#include <Vc/Vc>
#include <bla/matrix.h>
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <type_traits>

#include "utils.h"

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

template <typename E1, typename Func>
Matrix<scalar_type_t<E1>> apply(const MatrixExpression<E1> &e1, Func f) {
    Matrix<scalar_type_t<E1>> result({e1.rows(), e1.columns()});
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.columns(); ++j) {
            result[{i, j}] = f(e1[{i, j}]);
        }
    }

    return result;
}

template <typename E1, typename E2, typename Func>
Matrix<scalar_type_t<E1>> apply(const MatrixExpression<E1> &e1, const MatrixExpression<E2> &e2, Func f) {
    assert_eq(e1.size(), e2.size());
    Matrix<scalar_type_t<E1>> result({e1.rows(), e1.columns()});
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.columns(); ++j) {
            result[{i, j}] = f(e1[{i, j}], e2[{i, j}]);
        }
    }

    return result;
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
        ASSERT_EQ(m1 + m2, apply(m1, m2, [](int t1, int t2){ return t1 + t2; }));
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
        auto m3 = m1;

        m1 += m2;

        ASSERT_TRUE(isAlmostEqual(m1, apply(m3, m2, [](double t1, double t2){ return t1 + t2; })));
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
        auto m3 = m1;

        m1.sub({{0, 0}, {5, 76}}) += m2;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), apply(m3.sub({{0, 0}, {5, 76}}), m2, [](double t1, double t2){ return t1 + t2; })));
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
        ASSERT_TRUE(isAlmostEqual(m1 - m2, apply(m1, m2, [](double t1, double t2){ return t1 - t2; })));
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
        auto m3 = m1;

        m1 -= m2;

        ASSERT_TRUE(isAlmostEqual(m1, apply(m3, m2, [](double t1, double t2){ return t1 - t2; })));
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
        auto m3 = m1;

        m1.sub({{0, 0}, {5, 76}}) -= m2;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), apply(m3.sub({{0, 0}, {5, 76}}), m2, [](double t1, double t2){ return t1 - t2; })));
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
        auto m2 = m1;

        auto number = g();

        m1 *= number;

        ASSERT_TRUE(isAlmostEqual(m1, apply(m2, [number](double t1){ return t1 * number; })));
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
        auto m2 = m1;

        auto number = g();

        m1.sub({{0, 0}, {5, 76}}) *= number;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), apply(m2.sub({{0, 0}, {5, 76}}), [number](double t1){ return t1 * number; })));
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
        auto m2 = m1;

        auto number = g();

        m1 /= number;

        ASSERT_TRUE(isAlmostEqual(m1, apply(m2, [number](double t1){ return t1 / number; })));
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
        auto m2 = m1;

        auto number = g();

        m1.sub({{0, 0}, {5, 76}}) /= number;

        ASSERT_TRUE(isAlmostEqual(m1.sub({{0, 0}, {5, 76}}), apply(m2.sub({{0, 0}, {5, 76}}), [number](double t1){ return t1 / number; })));
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
        ASSERT_TRUE(isAlmostEqual(-m, apply(m, [](double t1){ return -t1; })));
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
        ASSERT_EQ(3. * m1, apply(m1, [](double t1){ return 3. * t1; }));
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
        ASSERT_EQ(m1 * 3., apply(m1, [](double t1){ return t1 * 3.; }));
    }
}

TEST(Operation, Multiple) {
    Matrix<double> m1({25, 25});
    Matrix<double> m2({25, 26});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        auto tmp1 = apply(m1, m1, [](double t1, double t2){ return t1 + t2; });
        auto tmp2 = apply(tmp1, [](double t1){ return -t1; });
        auto tmp3 = gemm(tmp2, m2);
        ASSERT_TRUE(isAlmostEqual(-(m1 + m1) * m2 + m2 - m2 + m2 - m2, tmp3));
    }
}

} // end namespace impl
} // namespace bla
