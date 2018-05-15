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
	if(a.size() != b.size()) { return false; }
	for(coordinate_type i = 0; i < a.rows(); ++i) {
		for(coordinate_type j = 0; j < a.columns(); ++j) {
			scalar_type_t<E1> diff = (a[{i, j}] - b[{i, j}]);
			if(diff * diff > epsilon) { return false; }
		}
	}
	return true;
}

template <typename E1, typename E2, typename T = double>
std::enable_if_t<std::is_same<scalar_type_t<E1>, std::complex<double>>::value, bool> isAlmostEqual(const MatrixExpression<E1>& a, const MatrixExpression<E2>& b,
                                                                                                   T epsilon = 0.001) {
	if(a.size() != b.size()) { return false; }
	for(coordinate_type i = 0; i < a.rows(); ++i) {
		for(coordinate_type j = 0; j < a.columns(); ++j) {
			scalar_type_t<E1> diff = (a[{i, j}] - b[{i, j}]);
			if(diff.real() * diff.real() > epsilon || diff.imag() * diff.imag() > epsilon) { return false; }
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

TEST(Operation, LUSolve) {
    Matrix<double> m1({3, 3});
    m1 << 3, 2, -1, 2, -2, 4, -1, 0.5, -1;

    auto lu = m1.LUDecomposition();

    Matrix<double> b({3, 1});

    b << 1, -2, 0;

    Matrix<double> x = lu.solve(b);

    std::cout << "x: " << x << std::endl;



    ASSERT_TRUE(isAlmostEqual(m1 * x, b));

}

TEST(Operation, QRDecomposition) {
	Matrix<double> m1({10, 5});

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&](const auto&) { return dis(gen); };
	for(int i = 0; i < 1; ++i) {
		m1.fill_seq(g);

		auto qr = m1.QRDecomposition();

		ASSERT_TRUE(isAlmostEqual(qr.getQ() * qr.getQ().transpose(), IdentityMatrix<double>(point_type{m1.rows(), m1.rows()})));

		ASSERT_TRUE(isAlmostEqual(m1, (qr.getQ() * qr.getR()).eval()));
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
