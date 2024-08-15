#include <bla/matrix.h>
#include <cstdlib>
#include <functional>
#include <gmpxx.h>

using namespace bla;

using data_type = mpf_class;

namespace bla {
namespace impl {

template <typename T>
struct operation_result<std::plus<>, T, data_type> : public detail::set_type<data_type> {};

template <typename T>
struct operation_result<std::plus<>, data_type, T> : public detail::set_type<data_type> {};

template <>
struct operation_result<std::plus<>, data_type, data_type> : public detail::set_type<data_type> {};

template <typename T>
struct operation_result<std::minus<>, T, data_type> : public detail::set_type<data_type> {};

template <typename T>
struct operation_result<std::minus<>, data_type, T> : public detail::set_type<data_type> {};

template <>
struct operation_result<std::minus<>, data_type, data_type> : public detail::set_type<data_type> {};

template <typename T>
struct operation_result<std::multiplies<>, T, data_type> : public detail::set_type<data_type> {};

template <typename T>
struct operation_result<std::multiplies<>, data_type, T> : public detail::set_type<data_type> {};

template <>
struct operation_result<std::multiplies<>, data_type, data_type> : public detail::set_type<data_type> {};

} // end namespace impl
} // namespace bla

int main() {
    Matrix<data_type> m1({10, 15});
    Matrix<data_type> m2({15, 7});
    Matrix<data_type> m3({10, 7});
    Matrix<data_type> m4({10, 7});

    m1.fill(2_mpf);

    m2.fill(1_mpf);

    m3.fill(4_mpf);

    m4.fill(7_mpf);

    Matrix<data_type> result = 3 * m3 + m1 * m2 + m4;

    return EXIT_SUCCESS;
}
