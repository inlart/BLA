#include <bla/matrix.h>
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
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

TEST(Evaluate, Vectorizable) {
    const coordinate_type s = 57;
    Matrix<double> m({s, s});
    Matrix<double> m1({s, s});
    Matrix<double> m2({s, s});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 10; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        // otherwise the test would be useless
        ASSERT_TRUE(vectorizable_v<decltype(m1 + m2)>);

        detail::evaluate_simplify(m1 + m2, m);


        ASSERT_TRUE(isAlmostEqual(m1 + m2, m));
    }
}

TEST(Evaluate, NotVectorizable) {
    struct A {
        A() : value(0.) {
        }

        A(double value) : value(value) {
        }

        A operator+(const A& other) const {
            return A{value + other.value};
        }

        A operator-(const A& other) const {
            return A{value - other.value};
        }

        A operator*(const A& other) const {
            return A{value * other.value};
        }

        bool operator>(const A& other) const {
            return value > other.value;
        }

        bool operator<(const A& other) const {
            return value < other.value;
        }

        double value;
    };

    const coordinate_type s = 57;
    Matrix<A> m({s, s});
    Matrix<A> m1({s, s});
    Matrix<A> m2({s, s});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 10; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        // otherwise the test would be useless
        ASSERT_FALSE(vectorizable_v<decltype(m1 + m2)>);

        detail::evaluate_simplify(m1 + m2, m);


        ASSERT_TRUE(isAlmostEqual(m1 + m2, m));
    }
}

TEST(Evaluate, Eval) {
    struct A {
        A() : value(0.) {
        }

        A(double value) : value(value) {
        }

        A operator+(const A& other) const {
            return A{value + other.value};
        }

        A operator-(const A& other) const {
            return A{value - other.value};
        }

        A operator*(const A& other) const {
            return A{value * other.value};
        }

        bool operator>(const A& other) const {
            return value > other.value;
        }

        bool operator<(const A& other) const {
            return value < other.value;
        }

        double value;
    };

    const coordinate_type s = 57;
    Matrix<A> m({s, s});
    Matrix<A> m1({s, s});
    Matrix<A> m2({s, s});


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };

    m.fill_seq(g);

    ASSERT_FALSE(std::is_reference<decltype((m1 + m2).eval())>::value);

    ASSERT_TRUE(isAlmostEqual(m1 + m2, (m1 + m2).eval()));
}

TEST(Evaluate, EvalVectorized) {
    const coordinate_type s = 57;
    Matrix<double> m({s, s});
    Matrix<double> m1({s, s});
    Matrix<double> m2({s, s});


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };

    m.fill_seq(g);

    ASSERT_FALSE(std::is_reference<decltype((m1 + m2).eval())>::value);

    ASSERT_TRUE(isAlmostEqual(m1 + m2, (m1 + m2).eval()));
}

TEST(Evaluate, EvalOptimize) {
    const coordinate_type s = 57;
    Matrix<double> m({s, s});

    MatrixExpression<Matrix<double>>& m_exp = m;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };

    m.fill_seq(g);

    ASSERT_EQ(std::addressof(m), std::addressof(m_exp.eval()));
}

} // end namespace impl
} // namespace bla
