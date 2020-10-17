#include <bla/matrix.h>
#include <cstdlib>

using namespace allscale::api::user::data;

using point_type = GridPoint<2>;
using data_type = double;

int main() {
    Matrix<data_type> m1({10, 15});
    Matrix<data_type> m2({15, 7});

    m1.fill(2);

    m2.fill(1);

    Matrix<data_type> m3 = m1 * m2;

    assert_eq(m3.size(), (point_type{m1.rows(), m2.columns()}));

    return EXIT_SUCCESS;
}
