#include <Vc/Vc>
#include <Vc/cpuid.h>
#include <allscale/api/user/data/matrix.h>
#include <gtest/gtest.h>
#include <iostream>

namespace allscale {
namespace api {
namespace user {
namespace data {

TEST(Matrix, Access) {
	Matrix<double> m({2, 2});
	m.zero();
	for(int i = 0; i < 2; ++i) {
		for(int j = 0; j < 2; ++j) {
			ASSERT_EQ(0.0, (m[{i, j}]));
		}
	}
}

TEST(Matrix, CustomTypes) {
    struct A;
    struct B;


    struct A {
        int operator+(const B&) const { return 1; }
        double operator-(const B&) const { return 0.1337; }
    };

    struct B {
        double operator+(const A&) const { return 0.1337; }
        int operator-(const A&) const { return 1; }
    };

    Matrix<A> m1({55, 58});
    Matrix<B> m2({55, 58});

    Matrix<int> m3({55, 58});
    Matrix<double> m4({55, 58});

    Matrix<int> test_i({55, 58});
    test_i.fill(1);

    Matrix<double> test_d({55, 58});
    test_d.fill(0.1337);

    m3 = m1 + m2;
    ASSERT_EQ(m3, test_i);

    m4 = m2 + m1;
    ASSERT_TRUE(isAlmostEqual(m4, test_d));

    m3 = m2 - m1;
    ASSERT_EQ(m3, test_i);

    m4 = m1 - m2;
    ASSERT_TRUE(isAlmostEqual(m4, test_d));
}

// -- utility
TEST(Utility, EigenConversion) {
	Matrix<int> m({4, 4});
	for(coordinate_type i = 0; i < m.rows(); ++i) {
		for(coordinate_type j = 0; j < m.columns(); ++j) {
			m[{i, j}] = i * m.columns() + j;
		}
	}
	Matrix<int> n(m.toEigenMatrix());

	ASSERT_EQ(m, n);
}

TEST(Utility, Random) {
	Matrix<double> m({2, 2});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	m.random(g);
	for(int i = 0; i < 2; ++i) {
		for(int j = 0; j < 2; ++j) {
			ASSERT_LE(-1.0, (m[{i, j}]));
			ASSERT_GE(+1.0, (m[{i, j}]));
		}
	}
}

TEST(Utility, Equal) {
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

TEST(Utility, EigenMap) {
    Matrix<double> m1({23, 45});
    Matrix<double> m2({m1.columns(), 53});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.random(g);
        m2.random(g);
        auto map1 = m1.getEigenMap();
        auto map2 = m2.getEigenMap();
        ASSERT_TRUE(isAlmostEqual(Matrix<double>(map1 * map2), Matrix<double>((m1.toEigenMatrix() * m2.toEigenMatrix()).eval())));
    }
}

TEST(Utility, Traits) {
    Matrix<double> m1({55, 56});
    Matrix<double> m2({55, 56});

    auto sum = m1 + m2;

    ASSERT_TRUE(vectorizable_v<decltype(sum)>);
    ASSERT_FALSE(vectorizable_v<decltype(m1 + m2.transpose())>);

    const Matrix<double> m3({55, 56});

    const volatile auto matrix_sum = m1 + m3;

    ASSERT_TRUE(vectorizable_v<decltype(matrix_sum)>);

    ASSERT_TRUE((std::is_same<double, scalar_type_t<decltype(matrix_sum)>>::value));

    Matrix<int> m4({55, 60});
    Matrix<int> m5({55, 60});

    ASSERT_TRUE((std::is_same<int, scalar_type_t<decltype(m4 + m5)>>::value));
    ASSERT_FALSE((std::is_same<double, scalar_type_t<decltype(m4 + m5)>>::value));
}

//-- expression
TEST(Expression, SubMatrix) {
    const int n = 8;
    const int nh = n / 2;
    Matrix<int> m1({n, n});

    algorithm::pfor(m1.size(), [&](const auto& p) { m1[p] = p.y % nh + nh * (p.x % nh); });

    Matrix<int> s1 = m1.sub({0, 0}, {nh, nh});
    Matrix<int> s2 = m1.sub({0, nh}, {nh, nh});
    Matrix<int> s3 = m1.sub({nh, 0}, {nh, nh});
    Matrix<int> s4 = m1.sub({nh, nh}, {nh, nh});

    ASSERT_EQ(s1, s2);
    ASSERT_EQ(s2, s3);
    ASSERT_EQ(s3, s4);

    s4[{0, 0}] = 1;

    ASSERT_NE(s4, s1);
}

TEST(Expression, IdentityMatrix) {
    Matrix<int> m1({37, 31});
    IdentityMatrix<int> m2(point_type{m1.columns(), m1.columns()});

    m1.fill(1337);

    Matrix<int> result(m1.size());

    result = m1 * m2;


    ASSERT_EQ(m1, result);
}

// -- operations
TEST(Operation, Addition) {
	Matrix<int> m1({123, 76});
	Matrix<int> m2(m1.size());
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, 10);

	auto g = [&]() { return dis(gen); };

	for(int i = 0; i < 20; ++i) {
		m1.random(g);
		m2.random(g);
		ASSERT_EQ(m1 + m2, Matrix<double>(m1.toEigenMatrix() + m2.toEigenMatrix()));
	}
}

TEST(Operation, AssignAddition) {
	Matrix<double> m1({123, 76});
	Matrix<double> m2(m1.size());
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 20; ++i) {
		m1.random(g);
		m2.random(g);

		Eigen::MatrixXd m1e = m1.toEigenMatrix();
		Eigen::MatrixXd m2e = m2.toEigenMatrix();

		m1 += m2;
		m1e += m2e;

		ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(m1e)));
	}
}

TEST(Operation, Subtraction) {
	Matrix<double> m1({31, 47});
	Matrix<double> m2(m1.size());
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 20; ++i) {
		m1.random(g);
		m2.random(g);
		ASSERT_TRUE(isAlmostEqual(m1 - m2, Matrix<double>(m1.toEigenMatrix() - m2.toEigenMatrix())));
		ASSERT_TRUE(isAlmostEqual(m1 - m1, m2 - m2));
	}
}

TEST(Operation, AssignSubtraction) {
	Matrix<double> m1({123, 76});
	Matrix<double> m2(m1.size());
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 20; ++i) {
		m1.random(g);
		m2.random(g);

		Eigen::MatrixXd m1e = m1.toEigenMatrix();
		Eigen::MatrixXd m2e = m2.toEigenMatrix();

		m1 -= m2;
		m1e -= m2e;

		ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(m1e)));
	}
}

TEST(Operation, Negation) {
	Matrix<double> m({100, 99});
	m.zero();
	ASSERT_TRUE(isAlmostEqual(m, -m));
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 20; ++i) {
		m.random(g);
		ASSERT_TRUE(isAlmostEqual(-m, Matrix<double>(-(m.toEigenMatrix()))));
	}
}

TEST(Operation, Multiplication) {
	Matrix<double> m1({45, 45});
	Matrix<double> m2({m1.columns(), 45});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 4; ++i) {
		m1.random(g);
		m2.random(g);
		ASSERT_TRUE(isAlmostEqual(m1 * m2, Matrix<double>((m1.toEigenMatrix() * m2.toEigenMatrix()).eval())));
	}
}

TEST(Operation, AssignMultiplication) {
	Matrix<double> m1({123, 76});
	Matrix<double> m2({m1.columns(), m1.columns()});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 20; ++i) {
		m1.random(g);
		m2.random(g);

		Eigen::MatrixXd m1e = m1.toEigenMatrix();
		Eigen::MatrixXd m2e = m2.toEigenMatrix();

		m1 *= m2;
		m1e *= m2e;

		ASSERT_TRUE(isAlmostEqual(m1, Matrix<double>(m1e)));
	}
}

TEST(Operation, MultiplicationStrassen) {
	Matrix<double> m1({8, 8});
	Matrix<double> m2({m1.columns(), 8});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 4; ++i) {
		m1.random(g);
		m2.random(g);
		ASSERT_TRUE(isAlmostEqual(strassen(m1, m2), Matrix<double>((m1.toEigenMatrix() * m2.toEigenMatrix()).eval())));
	}
}

TEST(Operation, MultiplicationBLAS) {
	Matrix<double> m1({231, 48});
	Matrix<double> m2({m1.columns(), 117});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 4; ++i) {
		m1.random(g);
		m2.random(g);

		Matrix<double> res({m1.rows(), m2.columns()});

		matrix_multiplication_blas(res, m1, m2);

		ASSERT_TRUE(isAlmostEqual(res, Matrix<double>((m1.toEigenMatrix() * m2.toEigenMatrix()).eval())));
	}
}

TEST(Operation, MultiplicationAllscale) {
	Matrix<double> m1({255, 127});
	Matrix<double> m2({m1.columns(), 84});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 4; ++i) {
		m1.random(g);
		m2.random(g);
		Matrix<double> m3({m1.rows(), m2.columns()});
		matrix_multiplication_allscale(m3, m1, m2);

		ASSERT_TRUE(isAlmostEqual(m3, Matrix<double>((m1.toEigenMatrix() * m2.toEigenMatrix()).eval())));
	}
}

TEST(Operation, MultiplicationAllscaleInteger) {
	Matrix<int> m1({255, 127});
	Matrix<int> m2({m1.columns(), 84});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, 10);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 4; ++i) {
		m1.random(g);
		m2.random(g);
		Matrix<int> m3({m1.rows(), m2.columns()});
		matrix_multiplication_allscale(m3, m1, m2);

		ASSERT_EQ(m3, Matrix<int>((m1.toEigenMatrix() * m2.toEigenMatrix()).eval()));
	}
}

TEST(Operation, ScalarMatrixMultiplication) {
	Matrix<double> m1({45, 45});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 4; ++i) {
		m1.random(g);
		ASSERT_EQ(3. * m1, Matrix<int>((3. * m1.toEigenMatrix()).eval()));
	}
}

TEST(Operation, MatrixScalarMultiplication) {
	Matrix<double> m1({45, 45});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	for(int i = 0; i < 4; ++i) {
		m1.random(g);
		ASSERT_EQ(m1 * 3., Matrix<int>((m1.toEigenMatrix() * 3.).eval()));
	}
}

TEST(Operation, Transpose) {
	Matrix<double> m1({47, 39});
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-1, 1);

	auto g = [&]() { return dis(gen); };
	m1.random(g);
	Matrix<double> m2 = m1.transpose();

	ASSERT_EQ(m1.rows(), m2.columns());
	ASSERT_EQ(m2.rows(), m1.columns());

	algorithm::pfor(m1.size(), [&](const point_type& p) { ASSERT_EQ(m1[p], (m2[{p.y, p.x}])); });
}

TEST(Operation, Multiple) {
    Matrix<double> m1({55, 55});
    Matrix<double> m2({55, 56});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.random(g);
        m2.random(g);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m1e = m1.toEigenMatrix();
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m2e = m2.toEigenMatrix();
        ASSERT_TRUE(isAlmostEqual(-(m1 + m1) * m2 + m2 - m2 + m2 - m2, Matrix<double>(-(m1e + m1e) * m2e + m2e - m2e + m2e - m2e)));
    }
}

TEST(Operation, DISABLED_Determinant) {
    Matrix<double> m1({2, 2});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.random(g);
        ASSERT_EQ(m1.determinant(), (m1[{0, 0}] * m1[{1, 1}] - m1[{0, 1}] * m1[{1, 0}]));
    }
}

TEST(Operation, DISABLED_DeterminantEigen) {
    Matrix<double> m1({41, 41});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&]() { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.random(g);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m1e = m1.toEigenMatrix();

        double det_diff = m1.determinant() - m1e.determinant();
        if(det_diff < 0) det_diff = -det_diff;
        ASSERT_TRUE(det_diff < 0.1);
    }
}

// -- simplify matrix expressions
TEST(Simplify, Transpose) {
	Matrix<int> m1({55, 58});
	Matrix<int> m2({55, 58});
	Matrix<int> m3({55, 58});
	Matrix<int> m4({55, 58});

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, 10);

	auto g = [&]() { return dis(gen); };

	m1.random(g);
	m2.random(g);
	m3.zero();
	m4.zero();

	m3 = m1 + m2;

	m4 = simplify(m1 + m2);

	ASSERT_EQ(m3, m4);

	m3 = m1.transpose().transpose();
	m4 = simplify(m1.transpose().transpose());

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

	auto g = [&]() { return dis(gen); };

	m1.random(g);
	m2.random(g);
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

	auto g = [&]() { return dis(gen); };

	m1.random(g);
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

    auto g = [&]() { return dis(gen); };

    m1.random(g);
    m2.random(g);

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

} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
