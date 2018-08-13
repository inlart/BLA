#include <Vc/Vc>
#include <allscale/api/user/data/matrix.h>
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <type_traits>

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
    Matrix<double> m1({571, 571});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1, 25);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 1; ++i) {
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

        algorithm::pfor(sv.getS().size(), [&](const auto& pos) {
            if(pos.x == pos.y) {
                ASSERT_GE(sv.getS()[pos], 0);
            } else {
                ASSERT_TRUE(std::abs(sv.getS()[pos]) < 10 * std::numeric_limits<double>::epsilon());
            }
        });


        ASSERT_TRUE(isAlmostEqual(m1, (sv.getU() * sv.getS() * sv.getV()).eval()));
    }
}

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
