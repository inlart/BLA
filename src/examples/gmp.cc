#include <allscale/api/user/data/matrix.h>
#include <cstdlib>
#include <gmpxx.h>

using namespace allscale::api::user::data;

using data_type = mpf_class;

namespace allscale {
namespace api {
namespace user {
namespace data {
namespace impl {

template <>
struct operation_result<std::plus<>, data_type, data_type> : public detail::set_type<data_type> {};

template <>
struct operation_result<std::minus<>, data_type, data_type> : public detail::set_type<data_type> {};

template <>
struct operation_result<std::multiplies<>, data_type, data_type> : public detail::set_type<data_type> {};

} // end namespace impl
} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale

int main() {
	Matrix<data_type> m1({10, 15});
	Matrix<data_type> m2({15, 7});
	Matrix<data_type> m3({10, 7});
	Matrix<data_type> m4({10, 7});

	m1.fill(2_mpf);

	m2.fill(1_mpf);

	m3.fill(4_mpf);

	m4.fill(7_mpf);

	Matrix<data_type> result = 3_mpf * m3 + m1 * m2 + m4;

	return EXIT_SUCCESS;
}
