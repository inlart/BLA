#include <Vc/Vc>
#include <bla/matrix.h>
#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <type_traits>

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

TEST(Vectorizable, Matrix) {
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

TEST(Vectorizable, UserDefinedType) {
    struct A {};

    Matrix<A> m1({55, 56});

    ASSERT_FALSE(vectorizable_v<decltype(m1)>);
}

TEST(Vectorizable, RefSubMatrix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1, 1);

    auto g = [&](const auto&) { return dis(gen); };

    for(int i = 0; i < 10; ++i) {
        Matrix<double> m1({55, 56});
        m1.fill_seq(g);
        const Matrix<double> m2(m1);

        auto refsub = m1.row(2);
        auto sub = m2.row(2);

        ASSERT_FALSE((std::is_same<decltype(refsub), decltype(sub)>::value));

        ASSERT_TRUE(vectorizable_v<decltype(refsub)>);
        ASSERT_TRUE(vectorizable_v<decltype(sub)>);

        ASSERT_TRUE(isAlmostEqual((m1.row(2) + m1.row(3)).eval(), (m2.row(2) + m2.row(3)).eval()));

        ASSERT_TRUE(vectorizable_v<decltype(m1.row(2) + m1.row(3))>);
        ASSERT_TRUE(vectorizable_v<decltype(m2.row(2) + m2.row(3))>);
    }
}

TEST(OperationResult, IntAddition) {
    ASSERT_TRUE((std::is_same<std::decay_t<operation_result_t<std::plus<>, int, int>>, int>::value));
}


TEST(OperationResult, UserDefinedAddition) {
    struct A {
        int x;
    };

    struct B {
        int x;

        B operator+(const A& a) {
            return B{a.x + this->x};
        }
    };

    ASSERT_TRUE((std::is_same<std::decay_t<operation_result_t<std::plus<>, B, A>>, B>::value));
}

TEST(ScalarType, Matrix) {
    const int size = 30;


    Matrix<double> m1{{size, size}};

    ASSERT_TRUE((std::is_same<scalar_type_t<decltype(m1)>, double>::value));
}

TEST(ScalarType, UserDefinedAddition) {
    const int size = 30;

    struct A {
        int x;
    };

    struct B {
        int x;

        B operator+(const A& a) {
            return B{a.x + this->x};
        }
    };

    Matrix<A> ma{{size, size}};
    Matrix<B> mb{{size, size}};

    ASSERT_TRUE((std::is_same<scalar_type_t<decltype(mb + ma)>, B>::value));
}

TEST(ExpressionMember, Matrix) {
    const int size = 30;

    Matrix<double> m1{{size, size}};

    using matrix_member = expression_member_t<decltype(m1)>;

    // we keep matrices as references
    ASSERT_TRUE((std::is_lvalue_reference<matrix_member>::value));
}


TEST(ExpressionMember, Addition) {
    const int size = 30;

    Matrix<double> m1{{size, size}};
    Matrix<double> m2{{size, size}};


    using add_member = expression_member_t<decltype(m1 + m2)>;

    // we copy MatrixExpressions
    ASSERT_FALSE((std::is_lvalue_reference<add_member>::value || std::is_rvalue_reference<add_member>::value));
}

TEST(TypeConsistent, IntAddition) {
    ASSERT_TRUE((type_consistent_v<std::plus<>, int>));
}

TEST(TypeConsistent, UserDefinedAddition) {
    struct A {
        int x;
    };

    struct B {
        int x;

        A operator+(const B& b) {
            return A{b.x + this->x};
        }
    };

    ASSERT_FALSE((type_consistent_v<std::plus<>, B>));
}

TEST(IsValid, Addition) {
    ASSERT_TRUE((is_valid_v<std::plus<>, int, int>));
    ASSERT_TRUE((is_valid_v<std::plus<>, int, double>));
}

TEST(IsValid, UserDefinedAdditionPlus) {
    struct A {};
    struct B {
        B operator+(const B&) {
            return B{};
        }
    };

    ASSERT_FALSE((is_valid_v<std::plus<>, A, A>));
    ASSERT_TRUE((is_valid_v<std::plus<>, B, B>));
}

} // end namespace impl
} // namespace bla
