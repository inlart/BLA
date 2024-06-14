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

TEST(Operation, Multiplication) {
    Matrix<double> m1({169, 45});
    Matrix<double> m2({m1.columns(), 97});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual((m1 * m2).eval(), gemm(m1, m2)));
    }
}


TEST(Operation, MultiplicationRefSub) {
    Matrix<double> m1({169, 45});
    Matrix<double> m2({m1.columns(), 97});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual((m1.sub({{13, 7}, {39, 13}}) * m2.sub({{3, 17}, {13, 27}})).eval(),
                                  gemm(m1.sub({{13, 7}, {39, 13}}), m2.sub({{3, 17}, {13, 27}}))));
    }
}

TEST(Operation, MultiplicationEval) {
    Matrix<double> m1({45, 45});
    Matrix<double> m2({m1.columns(), 45});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual((m1 * m2).eval(), gemm(m1, m2)));
    }
}

TEST(Operation, MultiplicationNoTransposeTranspose) {
    Matrix<double> m1({145, 43});
    Matrix<double> m2({59, m1.columns()});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual(m1 * m2.transpose(), gemm(m1, m2.transpose())));
    }
}

TEST(Operation, MultiplicationTransposeNoTranspose) {
    Matrix<double> m1({145, 43});
    Matrix<double> m2({m1.rows(), 59});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual(m1.transpose() * m2, gemm(m1.transpose(), m2)));
    }
}

TEST(Operation, MultiplicationTransposeTranspose) {
    Matrix<double> m1({145, 43});
    Matrix<double> m2({59, m1.rows()});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual(m1.transpose() * m2.transpose(), gemm(m1.transpose(), m2.transpose())));
    }
}

TEST(Operation, AssignMultiplication) {
    Matrix<double> m1({123, 76});
    Matrix<double> m2({m1.columns(), m1.columns()});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        auto m3 = m1;
        m1 *= m2;

        ASSERT_TRUE(isAlmostEqual(m1, gemm(m3, m2)));
    }
}

TEST(Operation, MultiplicationStrassen) {
    Matrix<double> m1({64, 64});
    Matrix<double> m2({m1.columns(), 64});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual(strassen<1>(m1, m2), gemm(m1, m2)));
    }
}

TEST(Operation, MultiplicationStrassenPadded) {
    Matrix<double> m1({9, 9});
    Matrix<double> m2({m1.columns(), 17});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        ASSERT_TRUE(isAlmostEqual(strassen<1>(m1, m2), gemm(m1, m2)));
    }
}

TEST(Operation, MultiplicationBLAS) {
    Matrix<double> m1({231, 48});
    Matrix<double> m2({m1.columns(), 117});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);

        Matrix<double> res({m1.rows(), m2.columns()});

        matrix_multiplication_blas(res, m1, m2);

        ASSERT_TRUE(isAlmostEqual(res, gemm(m1, m2)));
    }
}

TEST(Operation, MultiplicationFloat) {
    Matrix<float> m1({255, 127});
    Matrix<float> m2({m1.columns(), 84});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        Matrix<float> m3({m1.rows(), m2.columns()});
        m3 = m1 * m2;

        ASSERT_TRUE(isAlmostEqual(m3, gemm(m1, m2)));
    }
}

TEST(Operation, MultiplicationRowPermutation) {
    Matrix<double> m({19, 35});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    std::uniform_int_distribution<> dis_int(0, m.rows() - 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 10; ++i) {
        PermutationMatrix<double> p(m.rows());
        for(int k = 0; k < m.rows() / 4; ++k) {
            p.swap(dis_int(gen), dis_int(gen));
        }
        m.fill_seq(g);

        ASSERT_TRUE(isAlmostEqual(p * m, p.eval() * m));
    }
}

TEST(Operation, MultiplicationColumnPermutation) {
    Matrix<double> m({19, 35});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    std::uniform_int_distribution<> dis_int(0, m.rows() - 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 10; ++i) {
        PermutationMatrix<double> p(m.columns());
        for(int k = 0; k < m.rows() / 4; ++k) {
            p.swap(dis_int(gen), dis_int(gen));
        }
        m.fill_seq(g);

        ASSERT_TRUE(isAlmostEqual(m * p.transpose(), m * p.transpose().eval()));
    }
}

TEST(Operation, MultiplicationVectorVector) {
    const int n = 278;
    Matrix<double> a({n, n});
    Matrix<double> b({n, n});


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };

    a.fill_seq(g);
    b.fill_seq(g);

    int k = n / 2;

    ASSERT_TRUE(isAlmostEqual((a.column(k).bottomRows(n - k - 1) * b.row(k).bottomColumns(n - k - 1)).eval(),
                              gemm(a.column(k).bottomRows(n - k - 1), b.row(k).bottomColumns(n - k - 1))));
}

} // end namespace impl
} // namespace bla
