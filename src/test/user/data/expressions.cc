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

TEST(Matrix, Identity) {
    const coordinate_type s = 57;
    Matrix<double> m({s, s});
    m.identity();
    for(coordinate_type i = 0; i < s; ++i) {
        for(coordinate_type j = 0; j < s; ++j) {
            if(i == j)
                ASSERT_EQ(1.0, (m[{i, j}]));
            else
                ASSERT_EQ(0.0, (m[{i, j}]));
        }
    }
}

TEST(Matrix, Eye) {
    const point_type s{57, 68};
    Matrix<double> m(s);
    m.eye();
    for(coordinate_type i = 0; i < s.x; ++i) {
        for(coordinate_type j = 0; j < s.y; ++j) {
            if(i == j)
                ASSERT_EQ(1.0, (m[{i, j}]));
            else
                ASSERT_EQ(0.0, (m[{i, j}]));
        }
    }
}

TEST(Matrix, Norm) {
    const point_type s{256, 256};
    Matrix<double> m(s);
    m.identity();

    ASSERT_LT(std::abs(m.norm() - 16.), 1E-12);
}

TEST(Matrix, Max) {
    const point_type s{256, 256};
    Matrix<double> m(s);
    m.identity();

    ASSERT_EQ(m.max(), 1.);
}

TEST(Matrix, Min) {
    const point_type s{256, 256};
    Matrix<double> m(s);
    m.identity();

    ASSERT_EQ(m.min(), 0.);
}

TEST(Matrix, MaxElement) {
    for(int i = 0; i < 20; ++i) {
        const point_type s{256, 256};
        Matrix<int> m(s);
        m.identity();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, m.rows() * m.columns());

        int num = dis(gen);

        m[{num / m.columns(), num % m.columns()}] = 2;


        ASSERT_EQ(m.max_element(), m.begin() + num);
    }
}

TEST(Matrix, MinElement) {
    for(int i = 0; i < 20; ++i) {
        const point_type s{256, 256};
        Matrix<int> m(s);
        m.identity();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, m.rows() * m.columns());

        int num = dis(gen);

        m[{num / m.columns(), num % m.columns()}] = -1;


        ASSERT_EQ(m.min_element(), m.begin() + num);
    }
}

TEST(Matrix, Initializer) {
    const point_type s{4, 4};

    Matrix<int> m(s);

    m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;

    for(coordinate_type i = 0; i < m.rows(); ++i) {
        for(coordinate_type j = 0; j < m.columns(); ++j) {
            ASSERT_EQ((m[{i, j}]), (i * m.rows() + j + 1));
        }
    }
}

TEST(Matrix, Generator) {
    const point_type s{4, 4};
    Matrix<int> m1(s);
    Matrix<int> m2(s);

    m1 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;

    auto g = [s](const auto& pos) { return pos.x * s.x + pos.y; };

    m2.fill(g);

    ASSERT_EQ(m1, m2);
}

TEST(Matrix, CustomTypeInit) {
    struct A;
    struct B;


    struct A {
        A() : value(0) {
        }
        A(int x) : value(x) {
        } // needed to call the eye method - will be called with x = 0 and x = 1


        int operator+(const B&) const {
            return 1;
        }
        double operator-(const B&) const {
            return 0.1337 + value;
        }

    private:
        double value;
    };

    struct B {
        double operator+(const A&) const {
            return 0.1337;
        }
        int operator-(const A&) const {
            return 1;
        }
    };

    Matrix<A> m1({55, 58});
    Matrix<B> m2({55, 58});

    m1.eye();
}

TEST(Matrix, CustomTypes) {
    struct A;
    struct B;


    struct A {
        int operator+(const B&) const {
            return 1;
        }
        double operator-(const B&) const {
            return 0.1337;
        }
    };

    struct B {
        double operator+(const A&) const {
            return 0.1337;
        }
        int operator-(const A&) const {
            return 1;
        }
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

TEST(Matrix, Complex) {
    using type = std::complex<double>;

    Matrix<type> a({137, 239});
    Matrix<type> b({a.columns(), a.columns()});

    b.identity();

    algorithm::pfor(a.size(), [&](const auto& pos) { a[pos] = type(pos.x, pos.y); });

    ASSERT_TRUE(isAlmostEqual(a, Matrix<type>(a * b)));
}

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

    auto g = [&](const auto&) { return dis(gen); };
    m.fill_seq(g);
    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 2; ++j) {
            ASSERT_LE(-1.0, (m[{i, j}]));
            ASSERT_GE(+1.0, (m[{i, j}]));
        }
    }
}

TEST(Utility, EigenMap) {
    Matrix<double> m1({23, 45});
    Matrix<double> m2({m1.columns(), 53});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        auto map1 = m1.getEigenMap();
        auto map2 = m2.getEigenMap();
        ASSERT_TRUE(isAlmostEqual(Matrix<double>(map1 * map2), Matrix<double>((m1.toEigenMatrix() * m2.toEigenMatrix()).eval())));
    }
}

TEST(Expression, PermutationMatrix) {
    PermutationMatrix<int> p(5);

    ASSERT_TRUE(isAlmostEqual(Matrix<int>(p), Matrix<int>(IdentityMatrix<int>({5, 5}))));

    p.swap(0, 1);

    algorithm::pfor(p.size(), [&](const auto& pos) {
        if(pos.x == 0) {
            ASSERT_EQ(p[pos], pos.y == 1 ? 1 : 0);
        } else if(pos.x == 1) {
            ASSERT_EQ(p[pos], pos.y == 0 ? 1 : 0);
        } else {
            ASSERT_EQ(p[pos], pos.x == pos.y ? 1 : 0);
        }
    });
}

TEST(Expression, MatrixViewLower) {
    Matrix<double> m1({23, 45});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);

        algorithm::pfor(m1.size(), [&](const auto& pos) {
            if(pos.x >= pos.y)
                ASSERT_EQ(m1.view<ViewType::Lower>()[pos], m1[pos]);
            else
                ASSERT_EQ(m1.view<ViewType::Lower>()[pos], 0.);
        });
    }
}

TEST(Expression, MatrixViewUnitLower) {
    Matrix<double> m1({23, 45});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);

        algorithm::pfor(m1.size(), [&](const auto& pos) {
            if(pos.x > pos.y)
                ASSERT_EQ(m1.view<ViewType::UnitLower>()[pos], m1[pos]);
            else if(pos.x == pos.y)
                ASSERT_EQ(m1.view<ViewType::UnitLower>()[pos], 1.);
            else
                ASSERT_EQ(m1.view<ViewType::UnitLower>()[pos], 0.);
        });
    }
}

TEST(Expression, MatrixViewUpper) {
    Matrix<double> m1({23, 45});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);

        algorithm::pfor(m1.size(), [&](const auto& pos) {
            if(pos.x <= pos.y)
                ASSERT_EQ(m1.view<ViewType::Upper>()[pos], m1[pos]);
            else
                ASSERT_EQ(m1.view<ViewType::Upper>()[pos], 0.);
        });
    }
}

TEST(Expression, MatrixViewUnitUpper) {
    Matrix<double> m1({23, 45});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g);

        algorithm::pfor(m1.size(), [&](const auto& pos) {
            if(pos.x < pos.y)
                ASSERT_EQ(m1.view<ViewType::UnitUpper>()[pos], m1[pos]);
            else if(pos.x == pos.y)
                ASSERT_EQ(m1.view<ViewType::UnitUpper>()[pos], 1.);
            else
                ASSERT_EQ(m1.view<ViewType::UnitUpper>()[pos], 0.);
        });
    }
}

TEST(Expression, SubMatrix) {
    const int n = 8;
    const int nh = n / 2;
    Matrix<int> m1({n, n});

    algorithm::pfor(m1.size(), [&](const auto& p) { m1[p] = p.y % nh + nh * (p.x % nh); });

    Matrix<int> s1 = m1.sub({{0, 0}, {nh, nh}});
    Matrix<int> s2 = m1.sub({{0, nh}, {nh, nh}});
    Matrix<int> s3 = m1.sub({{nh, 0}, {nh, nh}});
    Matrix<int> s4 = m1.sub({{nh, nh}, {nh, nh}});

    ASSERT_EQ(s1, s2);
    ASSERT_EQ(s2, s3);
    ASSERT_EQ(s3, s4);

    s4[{0, 0}] = 1;

    ASSERT_NE(s4, s1);
}

TEST(Expression, RefSubMatrix) {
    const int n = 8;
    const int nh = n / 2;
    Matrix<int> m1({n, n});

    m1.fill(5);

    algorithm::pfor(m1.size(), [&](const auto& p) { ASSERT_EQ(m1[p], 5); });

    m1.sub({{0, 0}, {nh, nh}}).fill(1);
    m1.sub({{0, nh}, {nh, nh}}).fill(2);
    m1.sub({{nh, 0}, {nh, nh}}).fill(3);
    m1.sub({{nh, nh}, {nh, nh}}).fill(4);

    algorithm::pfor(m1.size(), [&](const auto& p) {
        if(p.x < nh && p.y < nh) {
            ASSERT_EQ(m1[p], 1);
        } else if(p.x < nh && p.y >= nh) {
            ASSERT_EQ(m1[p], 2);
        } else if(p.x >= nh && p.y < nh) {
            ASSERT_EQ(m1[p], 3);
        } else {
            ASSERT_EQ(m1[p], 4);
        }
    });
}

TEST(Expression, RefSubMatrixCopy) {
    Matrix<double> m1({123, 76});
    Matrix<double> m2({5, 76});

    m1.zero();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m2.fill_seq(g);

        auto m1_su = m1.topRows(5);

        m1_su = m2;

        algorithm::pfor(m1.size(), [&](const auto& pos) {
            if(pos < m1_su.size()) {
                ASSERT_EQ(m1[pos], m2[pos]);
            } else {
                ASSERT_EQ(m1[pos], 0);
            }
        });
    }
}

TEST(Expression, RefSubMatrixConversion) {
    Matrix<int> m1({197, 107});

    m1.fill(5);

    algorithm::pfor(m1.size(), [&](const auto& p) { ASSERT_EQ(m1[p], 5); });

    SubMatrix<Matrix<int>> m2 = m1;

    ASSERT_EQ(m1.size(), m2.size());

    algorithm::pfor(m1.size(), [&](const auto& p) { ASSERT_EQ(m1[p], m2[p]); });
}

TEST(Expression, RefSubMatrixSwap) {
    const int n = 38;
    const int nh = n / 2;
    Matrix<int> m1({n, n});

    m1.fill(5);

    algorithm::pfor(m1.size(), [&](const auto& p) { ASSERT_EQ(m1[p], 5); });

    m1.sub({{0, 0}, {nh, nh}}).fill(1);
    m1.sub({{0, nh}, {nh, nh}}).fill(2);
    m1.sub({{nh, 0}, {nh, nh}}).fill(3);
    m1.sub({{nh, nh}, {nh, nh}}).fill(4);

    m1.sub({{0, 0}, {nh, nh}}).swap(m1.sub({{nh, 0}, {nh, nh}}));


    algorithm::pfor(m1.size(), [&](const auto& p) {
        if(p.x < nh && p.y < nh) {
            ASSERT_EQ(m1[p], 3);
        } else if(p.x < nh && p.y >= nh) {
            ASSERT_EQ(m1[p], 2);
        } else if(p.x >= nh && p.y < nh) {
            ASSERT_EQ(m1[p], 1);
        } else {
            ASSERT_EQ(m1[p], 4);
        }
    });
}

TEST(Expression, RefSubMatrixContiguous) {
    const int n = 37;
    Matrix<int> m1({n, n});

    m1.fill(-1);

    algorithm::pfor(m1.size(), [&](const auto& p) { ASSERT_EQ(m1[p], -1); });

    for(int i = 0; i < n; ++i) {
        m1.row(i).fill(i);
    }

    //    ASSERT_TRUE(vectorizable_v<decltype(m1.row(0))>);

    algorithm::pfor(m1.size(), [&](const auto& p) { ASSERT_EQ(p.x, m1[p]); });
}

TEST(Expression, IdentityMatrix) {
    Matrix<int> m1({37, 31});
    IdentityMatrix<int> m2(point_type{m1.columns(), m1.columns()});

    m1.fill(1337);

    Matrix<int> result(m1.size());

    result = m1 * m2;


    ASSERT_EQ(m1, result);
}

TEST(Expression, Abs) {
    Matrix<int> m1({37, 31});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-10, -1);

    auto g = [&]() { return dis(gen); };

    m1.fill_seq(g);

    Matrix<int> m2 = m1.abs();

    algorithm::pfor(m1.size(), [&](const auto& pos) { ASSERT_EQ(m1[pos], -m2[pos]); });
}

TEST(Expression, Conjugate) {
    Matrix<double> m1({102, 53});
    Matrix<std::complex<double>> m2({102, 53});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g1 = [&](const auto&) { return dis(gen); };
    auto g2 = [&](const auto&) { return std::complex<double>(dis(gen), dis(gen)); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g1);
        m2.fill_seq(g2);

        ASSERT_TRUE(isAlmostEqual(m1.conjugate(), Matrix<double>(m1.toEigenMatrix().conjugate().eval())));
        ASSERT_TRUE(isAlmostEqual(m2.conjugate(), Matrix<std::complex<double>>(m2.toEigenMatrix().conjugate().eval())));
    }
}

TEST(Expression, Adjugate) {
    Matrix<double> m1({102, 53});
    Matrix<std::complex<double>> m2({102, 53});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g1 = [&](const auto&) { return dis(gen); };
    auto g2 = [&](const auto&) { return std::complex<double>(dis(gen), dis(gen)); };
    for(int i = 0; i < 4; ++i) {
        m1.fill_seq(g1);
        m2.fill_seq(g2);

        ASSERT_TRUE(isAlmostEqual(m1.adjoint(), Matrix<double>(m1.toEigenMatrix().adjoint().eval())));
        ASSERT_TRUE(isAlmostEqual(m2.adjoint(), Matrix<std::complex<double>>(m2.toEigenMatrix().adjoint().eval())));
    }
}

TEST(Expression, MatrixRowColumn) {
    Matrix<int> m1({37, 31});
    IdentityMatrix<int> m2(point_type{m1.columns(), m1.columns()});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);

        ASSERT_EQ(m1.row(15), m1.transpose().column(15).transpose());
    }
}

TEST(Expression, MatrixRow) {
    const coordinate_type c = 31;
    IdentityMatrix<int> m(point_type{c, c});

    for(int i = 0; i < c; ++i) {
        SubMatrix<IdentityMatrix<int>> r = m.row(i);
        for(int j = 0; j < c; ++j) {
            int val = i == j ? 1 : 0;
            ASSERT_EQ(val, (r[{0, j}]));
        }
    }
}

TEST(Expression, MatrixRowRange) {
    Matrix<int> m(point_type{31, 47});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 20; ++i) {
        m.fill_seq(g);

        range_type r = {4, 1};

        auto range = m.rowRange(r);

        for(int i = 0; i < r.y; ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                ASSERT_EQ((m[{i + r.x, j}]), (range[{i, j}]));
            }
        }

        range.fill(0);

        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                if(i >= r.x && i < r.x + r.y) {
                    ASSERT_EQ((m[{i, j}]), 0);
                } else {
                    ASSERT_NE((m[{i, j}]), 0);
                }
            }
        }
    }
}

TEST(Expression, MatrixColumnRange) {
    Matrix<int> m(point_type{31, 47});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 20; ++i) {
        m.fill_seq(g);

        range_type r = {4, 1};

        auto range = m.columnRange(r);

        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < r.y; ++j) {
                ASSERT_EQ((m[{i, j + r.x}]), (range[{i, j}]));
            }
        }

        range.fill(0);

        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                if(j >= r.x && j < r.x + r.y) {
                    ASSERT_EQ((m[{i, j}]), 0);
                } else {
                    ASSERT_NE((m[{i, j}]), 0);
                }
            }
        }
    }
}

TEST(Expression, MatrixTopRows) {
    Matrix<int> m(point_type{31, 47});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 20; ++i) {
        m.fill_seq(g);

        coordinate_type r = 4;

        auto range = m.topRows(r);

        ASSERT_EQ(range.size(), (point_type{r, m.columns()}));

        for(int i = 0; i < r; ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                ASSERT_EQ((m[{i, j}]), (range[{i, j}]));
            }
        }

        range.fill(0);

        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                if(i < r) {
                    ASSERT_EQ((m[{i, j}]), 0);
                } else {
                    ASSERT_NE((m[{i, j}]), 0);
                }
            }
        }
    }
}

TEST(Expression, MatrixBottomRows) {
    Matrix<int> m(point_type{31, 47});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 20; ++i) {
        m.fill_seq(g);

        coordinate_type r = 4;

        auto range = m.bottomRows(r);

        ASSERT_EQ(range.size(), (point_type{r, m.columns()}));

        for(int i = 0; i < r; ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                ASSERT_EQ((m[{m.rows() - r + i, j}]), (range[{i, j}]));
            }
        }

        range.fill(0);

        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                if(i >= m.rows() - r) {
                    ASSERT_EQ((m[{i, j}]), 0);
                } else {
                    ASSERT_NE((m[{i, j}]), 0);
                }
            }
        }
    }
}

TEST(Expression, MatrixTopColumns) {
    Matrix<int> m(point_type{31, 47});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 20; ++i) {
        m.fill_seq(g);

        coordinate_type r = 4;

        auto range = m.topColumns(r);

        ASSERT_EQ(range.size(), (point_type{m.rows(), r}));

        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < r; ++j) {
                ASSERT_EQ((m[{i, j}]), (range[{i, j}]));
            }
        }

        range.fill(0);

        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                if(j < r) {
                    ASSERT_EQ((m[{i, j}]), 0);
                } else {
                    ASSERT_NE((m[{i, j}]), 0);
                }
            }
        }
    }
}

TEST(Expression, MatrixBottomColumns) {
    Matrix<int> m(point_type{31, 47});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 20; ++i) {
        m.fill_seq(g);

        coordinate_type r = 4;

        auto range = m.bottomColumns(r);

        ASSERT_EQ(range.size(), (point_type{m.rows(), r}));

        for(int i = 0; i < r; ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                ASSERT_EQ((m[{i, m.columns() - r + j}]), (range[{i, j}]));
            }
        }

        range.fill(0);

        for(int i = 0; i < m.rows(); ++i) {
            for(int j = 0; j < m.columns(); ++j) {
                if(j >= m.columns() - r) {
                    ASSERT_EQ((m[{i, j}]), 0);
                } else {
                    ASSERT_NE((m[{i, j}]), 0);
                }
            }
        }
    }
}

TEST(Expression, MatrixColumn) {
    const coordinate_type c = 31;
    IdentityMatrix<int> m(point_type{c, c});

    for(int i = 0; i < c; ++i) {
        SubMatrix<IdentityMatrix<int>> r = m.column(i);
        for(int j = 0; j < c; ++j) {
            int val = i == j ? 1 : 0;
            ASSERT_EQ(val, (r[{j, 0}]));
        }
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

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m1e = m1.toEigenMatrix();

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

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m1e = m1.toEigenMatrix();

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

TEST(Operation, ElementMultiplication) {
    Matrix<double> m1({31, 47});
    Matrix<double> m2(m1.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };
    for(int i = 0; i < 20; ++i) {
        m1.fill_seq(g);
        m2.fill_seq(g);
        Matrix<double> m3(m1.product(m2));

        for(coordinate_type i = 0; i < m1.rows(); ++i) {
            for(coordinate_type j = 0; j < m1.rows(); ++j) {
                ASSERT_EQ((m3[{i, j}]), (m1[{i, j}] * m2[{i, j}]));
            }
        }
    }
}

TEST(Solve, ViewLower) {
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

TEST(Solve, ViewLowerInPlace) {
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

TEST(Solve, ViewLowerInverse) {
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

TEST(Solve, ViewUpper) {
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

TEST(Solve, ViewUpperInPlace) {
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

TEST(Solve, ViewUpperInverse) {
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

TEST(Solve, ViewUnitLower) {
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

TEST(Solve, ViewUnitLowerInPlace) {
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

TEST(Solve, ViewUnitLowerInverse) {
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

TEST(Solve, ViewUnitUpper) {
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

TEST(Solve, ViewUnitUpperInPlace) {
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

TEST(Solve, ViewUnitUpperInverse) {
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

TEST(Solve, EigenSolver) {
    Matrix<double> m1({4, 4});

    Matrix<double> zero({m1.rows(), 1});
    zero.zero();

    m1 << 52, 30, 49, 28, 30, 50, 8, 44, 49, 8, 46, 16, 28, 44, 16, 22;

    auto s = m1.solveEigen();

    for(unsigned int i = 0; i < s.eigenvalues.size(); ++i) {
        ASSERT_TRUE(isAlmostEqual(m1 * s.eigenvectors[i], s.eigenvalues[i] * s.eigenvectors[i], 0.01));

        // we are searching for non-trivial solutions
        ASSERT_FALSE(isAlmostEqual(s.eigenvectors[i], zero));
    }
}

} // end namespace impl
} // namespace data
} // namespace user
} // namespace api
} // namespace allscale
