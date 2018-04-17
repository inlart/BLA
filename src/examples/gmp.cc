#include <allscale/api/user/data/matrix.h>
#include <cstdlib>
#include <gmpxx.h>

using namespace allscale::api::user::data;

using data_type = mpf_class;

int main() {
	Matrix<data_type> m1({10, 15});
	Matrix<data_type> m2({15, 7});

	m1.fill(2);

	m2.fill(1);

	Matrix<data_type> result = m1 + m2;

	return EXIT_SUCCESS;
}
