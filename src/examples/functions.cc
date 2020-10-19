#include <bla/matrix.h>
#include <cstdlib>

using namespace bla;

// The function below will not compile because the copy constructor for MatrixExpression is protected.
// If it would be public the code would slice the expressions e1 and e2 and therefore most likely yield a segmentation fault.

// template <typename E1, typename E2>
// auto add(MatrixExpression<E1> e1, MatrixExpression<E2> e2) {
//	return e1 + e2;
//}


// This function takes the expression by reference which will not slice the object and works fine.

template <typename E1, typename E2>
auto add(const MatrixExpression<E1>& e1, const MatrixExpression<E2>& e2) {
    return e1 + e2;
}

using data_type = double;

int main() {
    Matrix<data_type> m1({10, 15});
    Matrix<data_type> m2({10, 15});
    Matrix<data_type> m3 = add(m1, m2);

    return EXIT_SUCCESS;
}
