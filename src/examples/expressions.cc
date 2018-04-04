#include <allscale/api/user/data/matrix.h>
#include <cstdlib>

using namespace allscale::api::user::data;

// template <typename E1, typename E2>
// auto add(MatrixExpression<E1> e1, MatrixExpression<E2> e2) {
//	return e1 + e2;
//}

template <typename E1, typename E2>
auto add(const MatrixExpression<E1>& e1, const MatrixExpression<E2>& e2) {
	return e1 + e2;
}

using data_type = double;

int main() {
	Matrix<data_type> m1({10, 15});
	Matrix<data_type> m2({10, 15});
	Matrix<data_type> m3({10, 15});

	m3 = add(m1, m2);

	return EXIT_SUCCESS;
}
