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

		algorithm::pfor(point_type{m1.size()},[&](const point_type& p) {
			ASSERT_EQ(m1[p], (m2[{p.y, p.x}]));
		});
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
