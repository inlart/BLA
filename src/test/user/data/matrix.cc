#include <gtest/gtest.h>
#include <allscale/api/user/data/matrix.h>


namespace allscale {
namespace api {
namespace user {
namespace data {

    TEST(Matrix, Access) {
        Matrix<double> m({2, 2});
        m.zero();
        for(int i = 0; i < 2; ++i) {
            for(int j = 0; j < 2; ++j) {
                ASSERT_EQ(0.0, (m[{i,j}]));
            }
        }
    }

    TEST(Matrix, Random) {
        Matrix<double> m({2, 2});
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
        	return dis(gen);
        };
        m.random(g);
        for(int i = 0; i < 2; ++i) {
            for(int j = 0; j < 2; ++j) {
                ASSERT_LE(-1.0, (m[{i,j}]));
                ASSERT_GE(+1.0, (m[{i,j}]));
            }
        }
    }

    TEST(Matrix, Addition) {
        Matrix<double> m1({123, 76});
        Matrix<double> m2(m1.size());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
        	return dis(gen);
        };
        for(int i = 0; i < 20; ++i) {
            m1.random(g);
            m2.random(g);
            ASSERT_TRUE(isAlmostEqual(m1+m2, Matrix<double>(m1.toEigenMatrix()+m2.toEigenMatrix())));
        }
    }

    TEST(Matrix, Subtraction) {
        Matrix<double> m1({31, 47});
        Matrix<double> m2(m1.size());
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
        	return dis(gen);
        };
        for(int i = 0; i < 20; ++i) {
            m1.random(g);
            m2.random(g);
            ASSERT_TRUE(isAlmostEqual(m1-m2, Matrix<double>(m1.toEigenMatrix()-m2.toEigenMatrix())));
            ASSERT_TRUE(isAlmostEqual(m1-m1, m2-m2));
        }
    }

    TEST(Matrix, Negation) {
        Matrix<double> m({100, 99});
        m.zero();
        ASSERT_TRUE(isAlmostEqual(m, -m));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
        	return dis(gen);
        };
        for(int i = 0; i < 20; ++i) {
            m.random(g);
            ASSERT_TRUE(isAlmostEqual(-m, Matrix<double>(-(m.toEigenMatrix()))));
        }
    }

    TEST(Matrix, Multiplication) {
        Matrix<double> m1({45, 45});
        Matrix<double> m2({(int)m1.columns(), 45});
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
        	return dis(gen);
        };
        for(int i = 0; i < 4; ++i) {
            m1.random(g);
            m2.random(g);
            ASSERT_TRUE(isAlmostEqual(m1*m2, Matrix<double>((m1.toEigenMatrix()*m2.toEigenMatrix()).eval())));
        }
    }

    TEST(Matrix, MultiplicationStrassen) {
        Matrix<double> m1({256, 256});
        Matrix<double> m2({m1.columns(), 256});
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
            return dis(gen);
        };
        for(int i = 0; i < 4; ++i) {
            m1.random(g);
            m2.random(g);
            ASSERT_TRUE(isAlmostEqual(strassen<1>(m1, m2), Matrix<double>((m1.toEigenMatrix()*m2.toEigenMatrix()).eval())));
        }
    }

    TEST(Matrix, ScalarMultiplication) {
        Matrix<int> m1({45, 45});
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-1, 1);

        auto g = [&]() {
            return dis(gen);
        };
        for(int i = 0; i < 4; ++i) {
            m1.random(g);
            ASSERT_EQ(3 * m1, Matrix<int>((3 * m1.toEigenMatrix()).eval()));
        }
    }

    TEST(Matrix, EigenMap) {
        Matrix<double> m1({23, 45});
        Matrix<double> m2({m1.columns(), 53});
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
            return dis(gen);
        };
        for(int i = 0; i < 4; ++i) {
            m1.random(g);
            m2.random(g);
            auto map1 = m1.getEigenMap();
            auto map2 = m2.getEigenMap();
            ASSERT_TRUE(isAlmostEqual( Matrix<double>(map1 * map2), Matrix<double>((m1.toEigenMatrix()*m2.toEigenMatrix()).eval())));
        }
    }

    TEST(Matrix, Equal) {
        Matrix<double> m1({45, 59});
        Matrix<double> m2({45, 59});

        for(int i = 0; i < 4; ++i) {
            m1.identity();
            m2.identity();
            ASSERT_EQ(m1, m2);
            ASSERT_EQ(m2, m1);

            m2 = 3. * m2;

            ASSERT_NE(m1, m2);
            ASSERT_NE(m2, m1);
        }
    }

    TEST(Matrix, PforEach) {
        Matrix<double> m1({50, 40});

        m1.zero();

        m1.pforEach([](auto& element) { element += 1; });

        for(int i = 0; i < m1.size()[0]; ++i) {
            for(int j = 0; j < m1.size()[1]; ++j) {
                ASSERT_EQ((m1[{i, j}]), 1);
            }
        }
    }

    TEST(Matrix, ForEach) {
        Matrix<double> m1({50, 40});

        m1.zero();

        int count = 0;

        m1.forEach([&](auto& element) { element += count; ++count; });

        for(int i = 0; i < m1.size()[0]; ++i) {
            for(int j = 0; j < m1.size()[1]; ++j) {
                ASSERT_EQ((m1[{i, j}]), (i * m1.size()[1] + j));
            }
        }
    }

    TEST(Matrix, Transpose) {
        Matrix<double> m1({47, 39});
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
            return dis(gen);
        };
        m1.random(g);
        Matrix<double> m2 = m1.transpose();

		ASSERT_EQ(m1.rows(), m2.columns());
		ASSERT_EQ(m2.rows(), m1.columns());

		algorithm::pfor(m1.size(),[&](const point_type& p) {
			ASSERT_EQ(m1[p], (m2[{p.y, p.x}]));
		});
    }

    TEST(Matrix, SubMatrix) {
        const int n = 8;
        const int nh = n / 2;
        Matrix<int> m1({n, n});

        algorithm::pfor(m1.size(), [&](const auto& p){
            m1[p] = p.y % nh + nh * (p.x % nh);
        });

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

    TEST(Matrix, MultipleOperations) {
        Matrix<double> m1({55, 55});
        Matrix<double> m2({55, 56});
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-1, 1);

        auto g = [&]() {
        	return dis(gen);
        };
        for(int i = 0; i < 20; ++i) {
            m1.random(g);
            m2.random(g);
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m1e = m1.toEigenMatrix();
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m2e = m2.toEigenMatrix();
            ASSERT_TRUE(isAlmostEqual(-(m1+m1)*m2 + m2 - m2 + m2 - m2, Matrix<double>(-(m1e+m1e)*m2e + m2e - m2e + m2e - m2e)));
        }
    }

} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
