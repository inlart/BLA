#include <bla/matrix.h>
#include <allscale/utils/assert.h>
#include <complex>
#include <cstdlib>

using namespace bla;

using data_type = std::complex<double>;

int main() {
    Matrix<data_type> m1({10, 15});
    Matrix<data_type> m2({15, 7});

    m1.fill({2, 3});

    m2.fill({3, 7});

    Matrix<data_type> m3 = m1 * m2;

    assert_eq(m3.size(), (point_type{m1.rows(), m2.columns()}));

    return EXIT_SUCCESS;
}
