#include <Eigen/Eigen>
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

TEST(Operation, LUDecomposition) {
    Matrix<double> m1({4, 4});
    m1 << 1.1, 1.2, 1.7, 3.2, 1.2, 2.7, 1.3, 1.7, 9.7, 2.7, 3.3, 2.1, 0.9, 2.1, 2.0, 1.5;

    auto lu = m1.LUDecomposition();

    ASSERT_TRUE(isAlmostEqual(lu.permutation() * m1, (lu.lower() * lu.upper()).eval()));
}

TEST(Operation, LUDecompositionBig) {
    Matrix<double> m1({17, 17});

    m1 << 4, 19, 10, 4, 5, 12, 2, 8, 12, 12, 25, 7, 22, 9, 4, 4, 20, 13, 18, 6, 13, 8, 13, 2, 5, 20, 8, 20, 15, 23, 1, 20, 22, 4, 12, 7, 9, 14, 5, 6, 7, 15, 15,
        1, 16, 24, 12, 24, 21, 25, 16, 9, 2, 21, 5, 11, 9, 13, 15, 15, 13, 14, 7, 5, 6, 22, 12, 10, 8, 19, 16, 24, 8, 7, 6, 14, 24, 23, 5, 5, 24, 2, 14, 9, 4,
        9, 14, 14, 21, 15, 25, 10, 14, 16, 20, 4, 6, 18, 24, 20, 24, 25, 17, 17, 24, 5, 25, 7, 1, 4, 23, 1, 13, 7, 13, 12, 16, 20, 7, 17, 12, 17, 18, 17, 18,
        19, 8, 16, 1, 25, 9, 16, 10, 16, 21, 19, 23, 5, 20, 11, 12, 20, 5, 18, 3, 14, 12, 3, 2, 25, 15, 3, 6, 7, 5, 12, 15, 6, 22, 22, 2, 10, 4, 13, 7, 25, 25,
        7, 12, 7, 21, 5, 21, 5, 8, 15, 17, 1, 13, 11, 15, 7, 7, 1, 4, 18, 17, 11, 13, 21, 9, 8, 8, 21, 15, 9, 1, 13, 13, 23, 5, 8, 6, 23, 14, 19, 14, 16, 15, 4,
        12, 6, 20, 11, 21, 19, 12, 23, 25, 18, 14, 22, 13, 7, 2, 15, 21, 17, 21, 3, 10, 1, 23, 25, 3, 5, 10, 25, 13, 16, 9, 2, 11, 25, 5, 3, 17, 4, 20, 8, 21,
        23, 12, 22, 18, 19, 24, 9, 23, 3, 9, 6, 11, 13, 2, 10, 7, 16, 9, 7, 8, 21, 1, 16, 18, 18, 1, 21, 17, 1, 10, 18, 17, 25, 8, 10, 1, 4, 1;

    auto lu = m1.LUDecomposition();

    ASSERT_TRUE(isAlmostEqual(lu.permutation() * m1, (lu.lower() * lu.upper()).eval()));
}

TEST(Operation, LUDecompositionRNG) {
    Matrix<double> m1({57, 57});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 25);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        auto lu = m1.LUDecomposition();

        ASSERT_TRUE(isAlmostEqual(lu.permutation() * m1, (lu.lower() * lu.upper()).eval()));
    }
}

TEST(Operation, LUSolve) {
    Matrix<double> m1({3, 3});
    m1 << 3, 2, -1, 2, -2, 4, -1, 0.5, -1;

    auto lu = m1.LUDecomposition();

    Matrix<double> b({3, 1});

    b << 1, -2, 0;

    Matrix<double> x = lu.solve(b);

    ASSERT_TRUE(isAlmostEqual(m1 * x, b));
}

TEST(Operation, FPLUDecompositionRNG) {
    Matrix<double> m1({5, 5});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 25);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        FPLUD<double> lu(m1);

        ASSERT_TRUE(isAlmostEqual(lu.rowPermutation() * m1 * lu.columnPermutation(), (lu.lower() * lu.upper()).eval()));
        ASSERT_TRUE(isAlmostEqual(lu.lower(), lu.lower().template view<ViewType::Lower>()));
        ASSERT_TRUE(isAlmostEqual(lu.upper(), lu.upper().template view<ViewType::Upper>()));
    }
}

TEST(Operation, FPLUDecompositionRank) {
    Matrix<double> m1({3, 3});

    m1 << 0, 1, 2, 1, 2, 1, 2, 7, 8;

    FPLUD<double> lu1(m1);

    ASSERT_EQ(lu1.rank(), 2);

    Matrix<double> m2({3, 3});

    m2 << 1, 0, 2, 2, 1, 0, 3, 2, 1;

    FPLUD<double> lu2(m2);

    ASSERT_EQ(lu2.rank(), 3);

    Matrix<double> m3({4, 4});

    m3 << 1, 4, 3, 2, 3, 7, 4, 6, 7, 8, 1, 14, 2, 11, 9, 4;

    FPLUD<double> lu3(m3);

    ASSERT_EQ(lu3.rank(), 2);
}

TEST(Operation, FPLUSolve) {
    Matrix<double> m1({3, 3});
    m1 << 3, 2, -1, 2, -2, 4, -1, 0.5, -1;

    auto lu = m1.FPLUDecomposition();

    Matrix<double> b({3, 1});

    b << 1, -2, 0;

    Matrix<double> x = lu.solve(b);

    ASSERT_TRUE(isAlmostEqual(m1 * x, b));
}

TEST(Operation, Householder) {
    Matrix<double> m1({57, 1});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        Householder<double> h({m1}, {m1.rows(), m1.rows()});

        ASSERT_TRUE(isAlmostEqual(h.getP() * h.getP(), IdentityMatrix<double>(point_type{m1.rows(), m1.rows()})));
        ASSERT_TRUE(isAlmostEqual(h.getP().inverse(), h.getP()));
    }
}

TEST(Operation, QRDecomposition) {
    Matrix<double> m1({5, 5});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 2);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        auto qr = m1.QRDecomposition();

        ASSERT_TRUE(isAlmostEqual(qr.getQ() * qr.getQ().transpose(), IdentityMatrix<double>(point_type{m1.rows(), m1.rows()})));

        ASSERT_TRUE(isAlmostEqual(m1, (qr.getQ() * qr.getR()).eval()));

        ASSERT_TRUE(isAlmostEqual(qr.getR(), qr.getR().template view<ViewType::Upper>()));
    }
}

TEST(Operation, DISABLED_SVDecomposition) {
    Matrix<double> m1({10, 5});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 1; ++i) {
        m1.fill_seq(g);

        auto sv = m1.SVDecomposition();

        ASSERT_TRUE(isAlmostEqual(sv.getU() * sv.getU().transpose(), IdentityMatrix<double>(point_type{sv.getU().rows(), sv.getU().rows()})));
        ASSERT_TRUE(isAlmostEqual(sv.getV().transpose() * sv.getV(), IdentityMatrix<double>(point_type{sv.getV().columns(), sv.getV().columns()})));

        ASSERT_GE(sv.getS().min(), 0);

        allscale::api::user::algorithm::pfor(sv.getS().size(), [&](const auto& pos) {
            if(pos.x == pos.y) {
                ASSERT_GE(sv.getS()[pos], 0);
            } else {
                ASSERT_TRUE(std::abs(sv.getS()[pos]) < 10 * std::numeric_limits<double>::epsilon());
            }
        });


        ASSERT_TRUE(isAlmostEqual(m1, (sv.getU() * sv.getS() * sv.getV()).eval()));
    }
}

TEST(Operation, Determinant) {
    Matrix<double> m1({2, 2});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        ASSERT_TRUE(std::abs(m1.determinant() - (m1[{0, 0}] * m1[{1, 1}] - m1[{0, 1}] * m1[{1, 0}])) < 0.0001);
    }
}

TEST(Operation, DeterminantEigen) {
    Matrix<double> m1({41, 41});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m1e = toEigenMatrix(m1);

        ASSERT_TRUE(std::abs(m1.determinant() - m1e.determinant()) < 0.001);
    }
}

TEST(Operation, DeterminantFPLUD) {
    Matrix<double> m1({2, 2});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        ASSERT_TRUE(std::abs(m1.FPLUDecomposition().determinant() - (m1[{0, 0}] * m1[{1, 1}] - m1[{0, 1}] * m1[{1, 0}])) < 0.0001);
    }
}

TEST(Operation, DeterminantFPLUDEigen) {
    Matrix<double> m1({41, 41});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m1e = toEigenMatrix(m1);

        ASSERT_TRUE(std::abs(m1.FPLUDecomposition().determinant() - m1e.determinant()) < 0.001);
    }
}

TEST(Operation, Inverse) {
    const point_type s{124, 124};
    Matrix<double> m1(s);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        Matrix<double> inv = m1.inverse();
        ASSERT_TRUE(isAlmostEqual(IdentityMatrix<double>(s), m1 * inv));
        ASSERT_TRUE(isAlmostEqual(IdentityMatrix<double>(s), inv * m1));
    }
}


TEST(Operation, InverseFPLUD) {
    const point_type s{4, 4};
    Matrix<double> m1(s);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        Matrix<double> inv = m1.FPLUDecomposition().inverse();
        ASSERT_TRUE(isAlmostEqual(IdentityMatrix<double>(s), m1 * inv));
        ASSERT_TRUE(isAlmostEqual(IdentityMatrix<double>(s), inv * m1));
    }
}

} // end namespace impl
} // namespace bla
