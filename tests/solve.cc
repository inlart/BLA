#include <bla/matrix.h>
#include <gtest/gtest.h>
#include <random>

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

TEST(View, Lower) {
    Matrix<double> m1({211, 211});
    Matrix<double> b({211, 5});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        b.fill_seq(g);

        auto x = m1.template view<ViewType::Lower>().solve(b);

        ASSERT_TRUE(isAlmostEqual(b, m1.template view<ViewType::Lower>() * x));
    }
}

TEST(View, LowerInPlace) {
    Matrix<double> m1({211, 211});
    Matrix<double> b({211, 5});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        b.fill_seq(g);

        Matrix<double> x = b;

        m1.template view<ViewType::Lower>().solveInPlace(x);

        ASSERT_TRUE(isAlmostEqual(b, m1.template view<ViewType::Lower>() * x));
    }
}

TEST(View, LowerInverse) {
    Matrix<double> m1({211, 211});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        auto x = m1.template view<ViewType::Lower>().inverse();

        ASSERT_TRUE(isAlmostEqual(IdentityMatrix<double>(m1.size()), m1.template view<ViewType::Lower>() * x));
    }
}

TEST(View, Upper) {
    Matrix<double> m1({211, 211});
    Matrix<double> b({211, 5});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        b.fill_seq(g);

        auto x = m1.template view<ViewType::Upper>().solve(b);

        ASSERT_TRUE(isAlmostEqual(b, m1.template view<ViewType::Upper>() * x));
    }
}

TEST(View, UpperInPlace) {
    Matrix<double> m1({211, 211});
    Matrix<double> b({211, 5});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        b.fill_seq(g);

        Matrix<double> x = b;

        m1.template view<ViewType::Upper>().solveInPlace(x);

        ASSERT_TRUE(isAlmostEqual(b, m1.template view<ViewType::Upper>() * x));
    }
}

TEST(View, UpperInverse) {
    Matrix<double> m1({211, 211});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        auto x = m1.template view<ViewType::Upper>().inverse();

        ASSERT_TRUE(isAlmostEqual(IdentityMatrix<double>(m1.size()), m1.template view<ViewType::Upper>() * x));
    }
}

TEST(View, UnitLower) {
    Matrix<double> m1({211, 211});
    Matrix<double> b({211, 5});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        b.fill_seq(g);

        auto x = m1.template view<ViewType::UnitLower>().solve(b);

        ASSERT_TRUE(isAlmostEqual(b, m1.template view<ViewType::UnitLower>() * x));
    }
}

TEST(View, UnitLowerInPlace) {
    Matrix<double> m1({211, 211});
    Matrix<double> b({211, 5});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        b.fill_seq(g);

        Matrix<double> x = b;

        m1.template view<ViewType::UnitLower>().solveInPlace(x);

        ASSERT_TRUE(isAlmostEqual(b, m1.template view<ViewType::UnitLower>() * x));
    }
}

TEST(View, UnitLowerInverse) {
    Matrix<double> m1({211, 211});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        auto x = m1.template view<ViewType::UnitLower>().inverse();

        ASSERT_TRUE(isAlmostEqual(IdentityMatrix<double>(m1.size()), m1.template view<ViewType::UnitLower>() * x));
    }
}

TEST(View, UnitUpper) {
    Matrix<double> m1({211, 211});
    Matrix<double> b({211, 5});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        b.fill_seq(g);

        auto x = m1.template view<ViewType::UnitUpper>().solve(b);

        ASSERT_TRUE(isAlmostEqual(b, m1.template view<ViewType::UnitUpper>() * x));
    }
}

TEST(View, UnitUpperInPlace) {
    Matrix<double> m1({211, 211});
    Matrix<double> b({211, 5});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        b.fill_seq(g);

        Matrix<double> x = b;

        m1.template view<ViewType::UnitUpper>().solveInPlace(x);

        ASSERT_TRUE(isAlmostEqual(b, m1.template view<ViewType::UnitUpper>() * x));
    }
}

TEST(View, UnitUpperInverse) {
    Matrix<double> m1({211, 211});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        auto x = m1.template view<ViewType::UnitUpper>().inverse();

        ASSERT_TRUE(isAlmostEqual(IdentityMatrix<double>(m1.size()), m1.template view<ViewType::UnitUpper>() * x));
    }
}

} // namespace impl
} // namespace bla
